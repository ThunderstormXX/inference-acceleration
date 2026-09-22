"""Scoped diagnostics of the pinned DFlash stream; production math is unchanged.

AST instrumentation only wraps existing statement groups and inserts drains.
No installed package is edited. Detailed module timings introduce many barriers
and must never be substituted for the separately recorded production latency.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack
import inspect
import textwrap
from typing import Any

from .recorder import ProfileRecorder, tensor_metadata
from ..backends.apple.speculative.dflash_backend import _temporary_attribute, _UPSTREAM_LOCK


def instrument_stream(function, profiler):
    """Fail closed if the pinned greedy stream's statement boundaries change."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)
             and ast.unparse(node.test) == 'n < max_tokens']
    if len(loops) != 1:
        raise ValueError('Expected exactly one DFlash decode loop')
    loop = loops[0]
    body = loop.body

    def locate(predicate):
        indices = [i for i, node in enumerate(body) if predicate(node)]
        if len(indices) != 1:
            raise ValueError('Pinned DFlash timing boundary changed')
        return indices[0]

    stream_blocks = [i for i, node in enumerate(body) if isinstance(node, ast.With)
                     and len(node.items) == 1
                     and ast.unparse(node.items[0].context_expr) == 'mx.stream(generation_stream)']
    if len(stream_blocks) != 2:
        raise ValueError('Expected exactly draft and verify stream contexts')
    draft = stream_blocks[0]
    verify = locate(lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == '_capture is not None')
    decision = locate(lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'd_list')
    emit = locate(lambda n: isinstance(n, ast.Expr) and isinstance(n.value, ast.Yield))
    rollback = locate(lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'trim')
    if not draft < verify < stream_blocks[1] < decision < emit < rollback or rollback != emit + 1:
        raise ValueError('Unexpected DFlash stage ordering')

    def wrap(name, statements):
        wrapper = ast.parse(f'with __dflash_profiler.stage({name!r}, locals()):\n    pass').body[0]
        drain = ast.parse(f'__dflash_profiler.drain({name!r}, locals())').body[0]
        wrapper.body = statements + [drain]
        return wrapper

    loop.body = (body[:draft] + [wrap('draft', body[draft:verify]),
                 wrap('verify', body[verify:decision]),
                 wrap('acceptance_commit', body[decision:emit]), body[emit],
                 wrap('rollback', body[rollback:])])
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    namespace['__dflash_profiler'] = profiler
    exec(compile(tree, inspect.getsourcefile(function) or '<dflash-profile>', 'exec'), namespace)
    return namespace[function.__name__]


@contextmanager
def _class_call(cls, replacement):
    owned = '__call__' in cls.__dict__
    previous = cls.__call__
    setattr(cls, '__call__', replacement)
    try:
        yield
    finally:
        if owned:
            setattr(cls, '__call__', previous)
        else:
            delattr(cls, '__call__')


class DFlashProfiler:
    """Phase attribution plus optional selected-round layer/operator diagnosis."""
    def __init__(self, backend, *, detail='phases', detail_rounds=(1, 10, 20)):
        if detail not in ('phases', 'layers', 'operators'):
            raise ValueError('detail must be phases, layers, or operators')
        self.backend = backend
        self.mx = backend._mx
        self.upstream = backend._upstream
        self.stream = self.upstream.generation_stream
        self.detail = detail
        self.detail_rounds = set(detail_rounds)
        self.round = 0
        self.phase = None
        self.event = None
        self.recorder = ProfileRecorder(
            synchronize=lambda: self.mx.synchronize(self.stream), evaluate=self.mx.eval)
        self._module_paths = {}
        self._module_stack = []
        self._gdn_index = 0

    @contextmanager
    def stage(self, name, state):
        if name == 'draft':
            self.round += 1
        self._gdn_index = 0
        metadata = {'kind': 'phase', 'round': self.round, 'phase': name,
                    'first_round': self.round == 1, 'emitted_before': state['n'],
                    'block_size': state['bs'], 'prefix_tokens': int(state['prompt'].size) + state['n'] - 1,
                    'context_hidden': tensor_metadata(state['hidden']),
                    'detail_enabled': self.detail != 'phases' and self.round in self.detail_rounds}
        old_phase, old_event = self.phase, self.event
        self.phase = name
        try:
            with self.mx.stream(self.stream), self.recorder.span(name, metadata) as event:
                self.event = event
                yield
        finally:
            self.phase, self.event = old_phase, old_event

    def drain(self, name, state):
        if name == 'draft':
            self.recorder.evaluate(state['draft_tokens'], [c.state for c in state['draft_cache']])
            self.event.metadata.update({'proposal_ids': state['draft_tokens'][0].tolist(),
                                        'draft_logits': tensor_metadata(state['draft_logits'])})
        elif name == 'verify':
            self.recorder.evaluate(state['target_tokens'], state['hidden'],
                                   [c.state for c in state['target_cache']])
            self.event.metadata.update({'verify_input': tensor_metadata(state['verify_input']),
                                        'logits': tensor_metadata(state['logits']),
                                        'target_ids': state['target_tokens'][0].tolist()})
        elif name == 'acceptance_commit':
            self.event.metadata.update({'accepted': state['accepted'], 'emitted': len(state['new_tokens']),
                                        'emitted_ids': list(state['new_tokens'])})
        elif name == 'rollback':
            self.recorder.evaluate(state['hidden'], [c.state for c in state['target_cache']])
            self.event.metadata.update({'accepted': state['accepted'], 'trimmed': state['trim']})

    def detailed(self):
        return self.phase is not None and self.detail != 'phases' and self.round in self.detail_rounds

    def _register_modules(self):
        # Register the actual layers even when upstream hides selected layers
        # behind its non-Module feature hooks. Shared head gets a stage alias.
        for actor, model in [('target', self.backend._model), ('draft', self.backend._draft)]:
            for path, module in model.named_modules():
                self._module_paths.setdefault(id(module), {})[actor] = (f'{actor}.{path}', module)
        for actor, layers in [('target', self.upstream._get_layers(self.backend._model)),
                              ('draft', self.backend._draft.layers)]:
            for index, layer in enumerate(layers):
                layer = getattr(layer, '_layer', layer)
                for path, module in layer.named_modules():
                    suffix = '.' + path if path else ''
                    self._module_paths.setdefault(id(module), {})[actor] = (
                        f'{actor}.layers.{index}{suffix}', module)

    def _module_call(self, original, module, args, kwargs):
        aliases = self._module_paths.get(id(module), {})
        actor = 'draft' if self.phase == 'draft' else 'target'
        entry = aliases.get(actor)
        if not self.detailed() or entry is None:
            return original(module, *args, **kwargs)
        path, _ = entry
        cls = type(module).__name__
        is_layer = path.split('.')[-1].isdigit() and '.layers.' in path
        if self.detail == 'layers' and not (is_layer or path.endswith(('.fc', '.lm_head'))):
            return original(module, *args, **kwargs)
        # GDN class itself is patched by upstream capture; its leaf modules and
        # the enclosing decoder remain instrumented, as does gated_delta_update.
        metadata = {'kind': 'module', 'phase': self.phase, 'round': self.round,
                    'class': cls, 'path': path, 'inputs': tensor_metadata(args),
                    'keyword_inputs': tensor_metadata(kwargs)}
        for key in ('weight', 'scales', 'biases'):
            if hasattr(module, key):
                metadata[key] = tensor_metadata(getattr(module, key))
        for key in ('bits', 'group_size', 'groups'):
            if hasattr(module, key):
                metadata[key] = getattr(module, key)
        if cls == 'QuantizedLinear':
            metadata['logical_weight_shape'] = [int(module.weight.shape[0]),
                                               int(module.scales.shape[1]) * int(module.group_size)]
        self.recorder.materialize(path + '.inputs', (args, kwargs),
                                  {'phase': self.phase, 'round': self.round})
        self._module_stack.append(path)
        try:
            return self.recorder.measured_call(path, original, module, *args,
                                               metadata=metadata, **kwargs)
        finally:
            self._module_stack.pop()

    def _operation_call(self, name, original, args, kwargs):
        if not self.detailed() or self.detail != 'operators':
            return original(*args, **kwargs)
        parent = self._module_stack[-1] if self._module_stack else self.phase
        if name == 'gated_delta_update':
            parent = f'{parent}.gdn_{self._gdn_index}'
            self._gdn_index += 1
        path = parent + '.' + name
        self.recorder.materialize(path + '.inputs', (args, kwargs),
                                  {'phase': self.phase, 'round': self.round})
        return self.recorder.measured_call(path, original, *args,
                    metadata={'kind': 'operation', 'operation': name, 'phase': self.phase,
                              'round': self.round, 'inputs': tensor_metadata(args),
                              'keyword_inputs': tensor_metadata(kwargs)}, **kwargs)

    @contextmanager
    def installed(self):
        with _UPSTREAM_LOCK, ExitStack() as stack:
            # Build the function AFTER the backend observer is installed: copied
            # globals must retain that observer's cache and acceptance hooks.
            original_observe = self.backend._observe
            @contextmanager
            def observed(observer):
                with original_observe(observer):
                    transformed = instrument_stream(self.upstream._stream_generate, self)
                    with _temporary_attribute(self.upstream, '_stream_generate', transformed):
                        yield
            stack.enter_context(_temporary_attribute(self.backend, '_observe', observed))
            self.backend._draft.bind(self.backend._model)
            self._register_modules()
            if self.detail != 'phases':
                classes = {type(module) for aliases in self._module_paths.values()
                           for _, module in aliases.values()}
                originals = {cls: cls.__call__ for cls in classes}
                for cls, original in originals.items():
                    def wrapped(module, *args, _original=original, **kwargs):
                        return self._module_call(_original, module, args, kwargs)
                    stack.enter_context(_class_call(cls, wrapped))
                if self.detail == 'operators':
                    for owner, name in [(self.mx.fast, 'scaled_dot_product_attention'),
                                        (self.upstream._gd_mod, 'gated_delta_update')]:
                        original = getattr(owner, name)
                        def wrapped_op(*args, _name=name, _original=original, **kwargs):
                            return self._operation_call(_name, _original, args, kwargs)
                        stack.enter_context(_temporary_attribute(owner, name, wrapped_op))
            yield self

    def measure(self, prompt_tokens, max_new_tokens):
        with self.installed():
            measurement = self.backend.measure(prompt_tokens, max_new_tokens)
        profile = self.recorder.to_dict()
        profile.update({'detail': self.detail, 'detail_rounds': sorted(self.detail_rounds),
                        'unattributed_decode_ms': measurement['decode_seconds'] * 1000 - profile['root_total_ms']})
        return {'measurement': measurement, 'profile': profile}


@contextmanager
def plain_target(backend):
    """Temporarily remove draft feature hooks for a normal AR baseline."""
    model = backend._model
    layers = backend._upstream._get_layers(model)
    old_layers = list(layers)
    has_hidden = hasattr(model, '_hidden_states')
    hidden = getattr(model, '_hidden_states', None)
    try:
        layers[:] = [getattr(layer, '_layer', layer) for layer in layers]
        if has_hidden:
            delattr(model, '_hidden_states')
        yield
    finally:
        layers[:] = old_layers
        if has_hidden:
            model._hidden_states = hidden
