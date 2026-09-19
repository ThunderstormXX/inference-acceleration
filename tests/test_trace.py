"""CPU-only checks for observable MTP events and stable replay text."""

from types import SimpleNamespace

import pytest

from inference_lab.visualization.trace import (
    EventRecorder,
    MTPTraceObserver,
    _patch_attribute,
    build_prompt,
    display_tokens,
)


def test_build_prompt_disables_thinking_and_requests_plain_token_ids():
    calls = []

    def template(messages, **kwargs):
        calls.append((messages, kwargs))
        return [3, 5, 8]

    assert build_prompt(SimpleNamespace(apply_chat_template=template), "A story.") == [3, 5, 8]
    assert calls == [(
        [{"role": "user", "content": "A story."}],
        {"tokenize": True, "return_dict": False,
         "add_generation_prompt": True, "enable_thinking": False},
    )]


def test_build_prompt_can_use_original_benchmark_thinking_mode():
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: calls.append(kw) or [3, 5])
    assert build_prompt(tokenizer, "A math problem.", enable_thinking=True) == [3, 5]
    assert calls[0]["enable_thinking"] is True


@pytest.mark.parametrize("tokens", [[], [True], [-1], [1.5], [[1]], {"input_ids": [1]}, (1,)])
def test_build_prompt_rejects_invalid_template_results(tokens):
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: tokens)
    with pytest.raises(ValueError, match="nonempty list of integer"):
        build_prompt(tokenizer, "story")


class ByteTokenizer:
    """A UTF-8 character split across three tokens, plus a hidden EOS."""

    pieces = {1: b"A ", 2: b"\xe2", 3: b"\x82", 4: b"\xac", 5: b"!", 6: b""}

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return b"".join(self.pieces[token] for token in tokens).decode("utf-8", errors="replace")


def test_display_tokens_delays_partial_utf8_and_preserves_final_text():
    result = display_tokens(ByteTokenizer(), [1, 2, 3, 4, 5, 6])
    assert result == {
        "output_text": "A €!",
        "token_texts": ["A ", "", "", "€", "!", ""],
        "decoded_prefixes": ["A ", "A ", "A ", "A €", "A €!", "A €!"],
    }
    assert "".join(result["token_texts"]) == result["output_text"]
    assert all(right.startswith(left) for left, right in zip(
        result["decoded_prefixes"], result["decoded_prefixes"][1:],
    ))


def test_display_tokens_handles_empty_output():
    assert display_tokens(ByteTokenizer(), []) == {
        "output_text": "", "token_texts": [], "decoded_prefixes": [],
    }


def test_display_tokens_never_retracts_text_when_prefix_decoder_revises():
    decoded = {(): "", (1,): "ab", (1, 2): "a?", (1, 2, 3): "abc"}
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: decoded[tuple(ids)])
    assert display_tokens(tokenizer, [1, 2, 3]) == {
        "output_text": "abc", "token_texts": ["ab", "", "c"],
        "decoded_prefixes": ["ab", "ab", "abc"],
    }


def test_patch_attribute_restores_local_value_after_exception():
    original = object()
    owner = SimpleNamespace(value=original)
    with pytest.raises(RuntimeError, match="failure"):
        with _patch_attribute(owner, "value", "temporary"):
            assert owner.value == "temporary"
            raise RuntimeError("failure")
    assert owner.value is original


def test_patch_attribute_restores_inherited_method_lookup_after_exception():
    class Base:
        def method(self):
            return "original"

    class Child(Base):
        pass

    owner = Child()
    with pytest.raises(RuntimeError, match="failure"):
        with _patch_attribute(owner, "method", lambda: "temporary"):
            assert owner.method() == "temporary"
            raise RuntimeError("failure")
    assert "method" not in vars(owner)
    assert owner.method.__func__ is Base.method
    assert owner.method() == "original"


class Array:
    def __init__(self, values, *, calls=None, label=None):
        self.values = list(values)
        self.calls = calls
        self.label = label

    def reshape(self, size):
        assert size == -1
        return self

    def tolist(self):
        if self.calls is not None:
            self.calls.append(f"{self.label} materialized")
        return list(self.values)


def fake_observer_round(accepted, output, *, fail=None, yielded=None):
    """Run the stock call sequence with plain Python values and no MLX import."""
    calls = []

    class Draft:
        def draft_block(self):
            calls.append("draft")
            return Array([11, 12], calls=calls, label="draft")

        def draft_eval_state(self):
            calls.append("draft state requested")
            return ["draft state"]

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "original decode"

    class VerifyResult:
        target_tokens = Array([11, 13, 14], calls=calls, label="target")
        hidden = "verified hidden"

        def commit(self, model, caches, count, block_size):
            calls.append(("commit", count, block_size))
            if fail == "commit":
                raise RuntimeError("commit failed")
            caches[0].state = "committed cache"
            return "commit result"

    def verify(*args, **kwargs):
        calls.append("verify")
        if fail == "verify":
            raise RuntimeError("verify failed")
        return VerifyResult()

    def walk(*args, **kwargs):
        calls.append("walk")
        return accepted, list(output)

    module = SimpleNamespace(
        _mtp_verify_target=verify, _mtp_acceptance_walk=walk,
        _MTPVerifyResult=VerifyResult,
    )

    class Backend:
        def _run_rounds(self):
            try:
                yield 10, "first"
                proposals = self._draft.draft_block()
                verification = module._mtp_verify_target(proposals)
                count, tokens = module._mtp_acceptance_walk(verification, proposals)
                cache = [SimpleNamespace(state="uncommitted cache")]
                assert verification.commit(None, cache, count, 3) == "commit result"
                for token in tokens if yielded is None else yielded:
                    yield token, None
            finally:
                calls.append("stream closed")

    backend = Backend()
    backend._draft = Draft()
    backend.tokenizer = Tokenizer()
    backend._mx = SimpleNamespace(
        eval=lambda *values: calls.append(("eval", values)),
        synchronize=lambda: calls.append("sync"),
    )
    ticks = iter(range(100, 1000))
    recorder = EventRecorder(clock=lambda: float(next(ticks)))
    observer = MTPTraceObserver(backend, recorder, module=module)
    originals = (Draft.draft_block, verify, walk, VerifyResult.commit, Backend._run_rounds, Tokenizer.decode)
    return backend, module, recorder, observer, calls, originals


def assert_restored(backend, module, originals):
    draft, verify, walk, commit, rounds, decode = originals
    assert "draft_block" not in vars(backend._draft)
    assert backend._draft.draft_block.__func__ is draft
    assert module._mtp_verify_target is verify
    assert module._mtp_acceptance_walk is walk
    assert module._MTPVerifyResult.commit is commit
    assert "_run_rounds" not in vars(backend)
    assert backend._run_rounds.__func__ is rounds
    assert "decode" not in vars(backend.tokenizer)
    assert backend.tokenizer.decode.__func__ is decode


@pytest.mark.parametrize("accepted,output,rejected,emitted_draft,emitted_target", [
    (0, [13], [11, 12], 0, 1),
    (1, [11, 13], [12], 1, 1),
    (2, [11, 12, 14], [], 2, 1),
    (2, [11, 12], [], 2, 0),
    (2, [11], [], 1, 0),
])
def test_mtp_observer_records_proposals_acceptance_rejections_and_clipped_output(
    accepted, output, rejected, emitted_draft, emitted_target,
):
    backend, module, recorder, observer, calls, originals = fake_observer_round(accepted, output)
    with observer.observe():
        assert backend.tokenizer.decode([10]) == ""
        recorder.start()
        yielded = list(backend._run_rounds())
        result = recorder.finish()

    assert yielded == [(10, "first"), *[(token, None) for token in output]]
    assert result["token_ids"] == observer.yielded == [10, *output]
    assert [event["type"] for event in result["events"]] == ["commit", "draft", "commit"]
    first, proposed, committed = result["events"]
    assert first["round"] == 0
    assert first["output_count"] == first["emitted_target_count"] == 1
    assert proposed["token_ids"] == [11, 12]
    assert committed["token_ids"] == output
    assert committed["output_count"] == len(output) + 1
    assert committed["accepted_count"] == accepted
    assert committed["draft_count"] == 2
    assert committed["round"] == proposed["round"] == 1
    assert committed["proposed_token_ids"] == [11, 12]
    assert committed["rejected_token_ids"] == rejected
    assert committed["target_token_ids"] == [11, 13, 14]
    assert committed["emitted_draft_count"] == emitted_draft
    assert committed["emitted_target_count"] == emitted_target
    assert first["t"] < proposed["t"] < committed["verification_completed_t"] < committed["t"]
    assert committed["cache_commit_enqueued_t"] == committed["cache_committed_t"] == committed["t"]
    assert "host enqueue" in committed["cache_commit_timing"]
    assert "lazy" in committed["cache_commit_timing"]
    assert result["prefill_seconds"] == first["t"]
    assert result["decode_seconds"] == result["total_seconds"] - first["t"]
    # Only first-token availability has an explicit observer barrier. Draft IDs
    # are materialized for display; target IDs wait until the native walk ends.
    assert calls == [
        "sync", "draft", "draft materialized", "verify", "walk",
        "target materialized", ("commit", accepted, 3), "stream closed",
    ]
    assert observer.pending is None
    assert_restored(backend, module, originals)


@pytest.mark.parametrize("failure", ["verify", "commit"])
def test_mtp_observer_restores_all_patches_and_closes_generator_on_failure(failure):
    backend, module, recorder, observer, calls, originals = fake_observer_round(1, [11, 13], fail=failure)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        with observer.observe():
            recorder.start()
            list(backend._run_rounds())
    assert recorder.token_ids == [10]
    assert calls[-1] == "stream closed"
    assert_restored(backend, module, originals)


def test_mtp_observer_rejects_generator_output_that_disagrees_with_committed_ids():
    backend, module, recorder, observer, calls, originals = fake_observer_round(1, [11, 13], yielded=[99])
    with pytest.raises(RuntimeError, match="event output disagrees"):
        with observer.observe():
            recorder.start()
            list(backend._run_rounds())
    assert calls[-1] == "stream closed"
    assert_restored(backend, module, originals)
