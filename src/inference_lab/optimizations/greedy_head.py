"""Opt-in fused vocabulary argmax for greedy Qwen3.5 AR and pinned DFlash.

The full vocabulary is still evaluated. The kernel avoids materializing full
logits by returning per-tile winners followed by a reduction. This is an
execution experiment: exact token parity must be measured, never presumed.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager, ExitStack
from hashlib import sha256
import inspect
from pathlib import Path
import textwrap

from .affine import load_affine_verifier
from ..backends.apple.speculative.dflash_backend import _UPSTREAM_LOCK


_REPLACEMENTS = {
    "draft_logits = draft(block, hidden, draft_cache, logits_start=1)":
        "draft_hidden = draft.hidden_states(block, hidden, draft_cache, logits_start=1)",
    "draft_tokens = mx.argmax(draft_logits, axis=-1)":
        "draft_tokens = __greedy_head_adapter.draft_argmax(draft, draft_hidden)",
    "logits = model(verify_input, target_cache)":
        "verify_hidden = __greedy_head_adapter.target_hidden(model, verify_input, target_cache)",
    "target_tokens = mx.argmax(logits, axis=-1)":
        "target_tokens = __greedy_head_adapter.target_argmax(model, verify_hidden)",
}


@contextmanager
def _scoped_attribute(owner, name, value):
    """Restore the original instance/class lookup, including inherited methods."""
    owned = name in vars(owner)
    previous = vars(owner).get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def transform_greedy_stream(function, adapter):
    """Replace exactly four pinned statements and preserve observer globals."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected a single pinned DFlash stream function")
    definition = tree.body[0]
    if definition.name != "_stream_generate":
        raise ValueError("Expected the pinned _stream_generate function")
    if definition.decorator_list:
        raise ValueError("Decorated DFlash stream is not supported")
    parameters = {arg.arg for arg in definition.args.args + definition.args.kwonlyargs}
    if not {"model", "draft", "temperature", "max_tokens"} <= parameters:
        raise ValueError("Pinned DFlash stream signature changed")
    loops = [node for node in ast.walk(definition) if isinstance(node, ast.While)
             and ast.unparse(node.test) == "n < max_tokens"]
    if len(loops) != 1:
        raise ValueError("Expected one pinned DFlash decode loop")
    loop = loops[0]
    assignments = [node for node in ast.walk(loop) if isinstance(node, ast.Assign)]
    matches = {text: [node for node in assignments if ast.unparse(node) == text]
               for text in _REPLACEMENTS}
    if any(len(nodes) != 1 for nodes in matches.values()):
        raise ValueError("Pinned DFlash greedy assignments changed; expected exactly four matches")
    # Counting writes as well as matching text catches added alternative writes
    # that would silently override a fused result in a newer upstream stream.
    expected_counts = {"draft_logits": 1, "logits": 1, "target_tokens": 1}
    for name, count in expected_counts.items():
        actual = sum(isinstance(target, ast.Name) and target.id == name
                     for node in assignments for target in node.targets)
        if actual != count:
            raise ValueError(f"Unexpected additional DFlash writes to {name}")
    if not (matches[next(iter(_REPLACEMENTS))][0].lineno
            < matches["draft_tokens = mx.argmax(draft_logits, axis=-1)"][0].lineno
            < matches["logits = model(verify_input, target_cache)"][0].lineno
            < matches["target_tokens = mx.argmax(logits, axis=-1)"][0].lineno):
        raise ValueError("Pinned DFlash greedy statement order changed")

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            replacement = _REPLACEMENTS.get(ast.unparse(node))
            return ast.copy_location(ast.parse(replacement).body[0], node) if replacement else node

    tree = Rewrite().visit(tree)
    guard = ast.parse("__greedy_head_adapter.validate_stream(model, draft, temperature)").body[0]
    definition.body.insert(0, guard)
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    namespace["__greedy_head_adapter"] = adapter
    filename = inspect.getsourcefile(function) or "<greedy-head-experiment>"
    exec(compile(tree, filename, "exec"), namespace)
    adapter.stream_source_sha256 = sha256(source.encode()).hexdigest()
    adapter.transformed_source_sha256 = sha256(ast.unparse(tree).encode()).hexdigest()
    return namespace[definition.name]


class ScopedGreedyHead:
    """Restore all AR and DFlash overrides when the experiment scope closes."""

    def __init__(self, backend, verifier_module=None):
        self.backend = backend
        if verifier_module is None:
            self.verifier, self.source_metadata = load_affine_verifier()
        else:
            self.verifier = verifier_module
            self.source_metadata = {"import_method": "provided verifier module"}
            source_file = getattr(verifier_module, "__file__", None)
            if source_file and Path(source_file).is_file():
                self.source_metadata.update(verifier_file=str(Path(source_file).resolve()),
                    verifier_source_sha256=sha256(Path(source_file).read_bytes()).hexdigest())
        if not callable(getattr(self.verifier, "optimized_affine_argmax", None)):
            raise ValueError("Verifier must expose optimized_affine_argmax")
        if backend._model is None or backend._mx is None:
            raise ValueError("Load the backend before constructing the greedy-head context")
        self._target_parts(backend._model)
        self.upstream = getattr(backend, "_upstream", None)
        self.draft = getattr(backend, "_draft", None)
        if self.upstream is not None:
            if self.draft is None:
                raise ValueError("DFlash backend must have a loaded drafter")
            self.validate_stream(backend._model, self.draft, 0.0)
        self.hits = Counter()
        self.fallbacks = Counter()
        self.stock_ar_prefill_calls = 0
        self.stream_source_sha256 = None
        self.transformed_source_sha256 = None
        self._prefill_pending = False
        self._stack = None

    @staticmethod
    def _target_parts(model):
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            raise ValueError("Greedy head adapter requires the Qwen3.5 language_model wrapper")
        if getattr(getattr(language_model, "args", None), "tie_word_embeddings", False):
            raise ValueError("Tied target embeddings are unsupported by this experiment")
        decoder = getattr(language_model, "model", None)
        head = getattr(language_model, "lm_head", None)
        if not callable(decoder) or not callable(head):
            raise ValueError("Qwen3.5 target must expose its decoder and explicit LM head")
        return decoder, head

    def validate_stream(self, model, draft, temperature):
        if temperature != 0.0:
            raise ValueError("Fused greedy head requires temperature=0")
        if model is not self.backend._model or draft is not self.draft:
            raise ValueError("Greedy head context is restricted to its loaded target and drafter")
        draft2 = getattr(self.upstream, "DFlash2DraftModel", None)
        if (draft2 is not None and isinstance(draft, draft2)) or type(draft).__name__ == "DFlash2DraftModel":
            raise ValueError("DFlash2 candidate selection is unsupported by this adapter")
        config = getattr(draft, "config", None)
        if config is None or getattr(config, "output_multiplier", None) != 1.0:
            raise ValueError("Drafter output_multiplier must be exactly 1")
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is not None and not softcap <= 0:
            raise ValueError("Positive or non-finite drafter logit softcapping is unsupported")
        if not callable(getattr(draft, "hidden_states", None)) or not callable(getattr(draft, "compute_logits", None)):
            raise ValueError("Expected ordinary DFlash hidden_states and compute_logits methods")
        self._target_parts(model)

    def target_hidden(self, model, inputs, cache):
        decoder, _ = self._target_parts(model)
        # The same decoder still runs DFlash's feature hooks and GDN capture.
        return decoder(inputs, cache)

    def _argmax(self, head, hidden, *, actor, fallback):
        tokens = int(hidden.shape[1])
        key = f"{actor}:T{tokens}"
        result = self.verifier.optimized_affine_argmax(head, hidden)
        if result is not None:
            self.hits[key] += 1
            return result
        self.fallbacks[key] += 1
        return self.backend._mx.argmax(fallback(hidden), axis=-1)

    def target_argmax(self, model, hidden, *, actor="target_verify"):
        _, head = self._target_parts(model)
        return self._argmax(head, hidden, actor=actor, fallback=head)

    def draft_argmax(self, draft, hidden):
        return self._argmax(draft.lm_head, hidden, actor="draft", fallback=draft.compute_logits)

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("The same greedy-head context cannot be entered twice")
        stack = ExitStack()
        stack.enter_context(_UPSTREAM_LOCK)
        try:
            original_next = self.backend._next_token
            original_measure = self.backend._measure

            def next_token(input_tokens, cache):
                if self._prefill_pending:
                    self._prefill_pending = False
                    self.stock_ar_prefill_calls += 1
                    return original_next(input_tokens, cache)
                hidden = self.target_hidden(self.backend._model, input_tokens[None], cache)
                self.backend._quantize_cache(cache)
                return self.target_argmax(self.backend._model, hidden[:, -1:], actor="ar")[:, 0]

            def measure(*args, **kwargs):
                previous = self._prefill_pending
                self._prefill_pending = True
                try:
                    return original_measure(*args, **kwargs)
                finally:
                    self._prefill_pending = previous

            stack.enter_context(_scoped_attribute(self.backend, "_next_token", next_token))
            stack.enter_context(_scoped_attribute(self.backend, "_measure", measure))
            if self.upstream is not None:
                original_observe = self.backend._observe

                @contextmanager
                def observe(observer):
                    # Copy globals after observer installs its cache/rollback
                    # hooks, exactly as the existing scoped profiler does.
                    with original_observe(observer):
                        transformed = transform_greedy_stream(self.upstream._stream_generate, self)
                        with _scoped_attribute(self.upstream, "_stream_generate", transformed):
                            yield

                stack.enter_context(_scoped_attribute(self.backend, "_observe", observe))
            self._stack = stack
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, exc_type, exc, traceback):
        stack, self._stack = self._stack, None
        return stack.__exit__(exc_type, exc, traceback)

    def reset_counters(self):
        self.hits.clear()
        self.fallbacks.clear()
        self.stock_ar_prefill_calls = 0

    def metadata(self):
        return {"optimization": "full-vocabulary fused greedy head",
                "optimized_argmax_calls": dict(self.hits),
                "fallback_argmax_calls": dict(self.fallbacks),
                "stock_ar_prefill_calls": self.stock_ar_prefill_calls,
                "dflash_prefill_policy": "unchanged original upstream prefill",
                "ar_prefill_policy": "unchanged first _next_token per measured request",
                "dflash_stream_source_sha256": self.stream_source_sha256,
                "dflash_transformed_source_sha256": self.transformed_source_sha256,
                "dflash_assignment_replacements": len(_REPLACEMENTS),
                "numerical_parity": "not assumed; compare generated token IDs",
                **self.source_metadata}
