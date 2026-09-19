"""Timing accounting checks that do not import MLX or allocate a GPU model."""

from types import SimpleNamespace
import unittest

from inference_lab.backends.apple.mlx_backend import MLXBackend
from inference_lab.backends.apple.mlx_vlm_backend import MLXVLMBackend


class FakeArray:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, key):
        return self if key is None else FakeArray(self.values[key])

    def item(self):
        assert len(self.values) == 1
        return self.values[0]


class FakeLogits:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self


class FakeDevice:
    uint32 = "uint32"

    def __init__(self):
        self.events = []

    def array(self, values, dtype=None):
        return FakeArray(values)

    def eval(self, *arrays):
        self.events.append("eval")

    def async_eval(self, *arrays):
        self.events.append("async_eval")

    def synchronize(self):
        self.events.append("synchronize")

    def clear_cache(self):
        pass

    def reset_peak_memory(self):
        pass

    def get_peak_memory(self):
        return 123_000_000

    def argmax(self, logits, axis):
        return FakeArray([logits.value])


class MLXMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.backend = MLXBackend("/tmp/fake-model", prefill_step_size=2)
        self.backend._mx = FakeDevice()
        self.backend._tokenizer = SimpleNamespace(
            decode=lambda values, **kwargs: ",".join(map(str, values))
        )
        self.caches = []
        self.forwards = []

        def new_cache(model):
            cache = [SimpleNamespace(state=[])]
            self.caches.append(cache)
            return cache

        def model(tokens, cache):
            self.forwards.append((tokens.values, cache))
            self.backend._mx.events.append(("forward", tuple(tokens.values)))
            return FakeLogits(tokens.values[-1] + 1)

        self.backend._make_prompt_cache = new_cache
        self.backend._model = model

    def test_no_decode_forward_leaks_into_prefill_or_past_output_limit(self):
        result = self.backend.measure([1, 2, 3, 4, 5], max_new_tokens=4)
        self.assertEqual(result["generated_token_ids"], [6, 7, 8, 9])
        self.assertEqual(result["decode_tokens"], 3)
        self.assertEqual(result["prompt_tokens"], 5)
        self.assertEqual(
            [tokens for tokens, _ in self.forwards],
            [[1, 2], [3, 4], [5], [6], [7], [8]],
        )
        events = self.backend._mx.events
        final_prompt = events.index(("forward", (5,)))
        first_decode = events.index(("forward", (6,)))
        self.assertIn("synchronize", events[final_prompt:first_decode])
        self.assertEqual(events[-1], "synchronize")
        self.assertEqual(result["peak_memory_gb"], 0.123)

    def test_single_generated_token_has_no_decode_phase(self):
        result = self.backend.measure([7], max_new_tokens=1)
        self.assertEqual(result["generated_token_ids"], [8])
        self.assertEqual(result["decode_tokens"], 0)
        self.assertEqual(result["decode_seconds"], 0.0)
        self.assertEqual(len(self.forwards), 1)
        self.assertNotIn("async_eval", self.backend._mx.events)

    def test_independent_requests_do_not_reuse_cache(self):
        self.backend.measure([7], max_new_tokens=2)
        self.backend.measure([7], max_new_tokens=2)
        self.assertIsNot(self.caches[0], self.caches[1])
        self.assertIs(self.forwards[0][1], self.caches[0])
        self.assertIs(self.forwards[1][1], self.caches[0])
        self.assertIs(self.forwards[2][1], self.caches[1])
        self.assertIs(self.forwards[3][1], self.caches[1])

    def test_invalid_token_workloads_are_rejected(self):
        for tokens, n in [([], 1), ([-1], 1), ([True], 1), ([1], 0), ([1], 1.5)]:
            with self.subTest(tokens=tokens, n=n):
                with self.assertRaises(ValueError):
                    self.backend.measure(tokens, n)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"prefill_step_size": 0}, {"kv_bits": 3}, {"kv_bits": 4.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MLXBackend("/tmp/fake-model", **kwargs)

    def test_vlm_resets_rotary_state_and_unwraps_logits(self):
        backend = MLXVLMBackend("/tmp/fake-model", prefill_step_size=2)
        backend._mx = self.backend._mx
        backend._tokenizer = self.backend._tokenizer
        backend._make_prompt_cache = self.backend._make_prompt_cache
        positions_at_start = []

        class LanguageModel:
            _position_ids = "old positions"
            _rope_deltas = "old deltas"

            def __call__(model, tokens, cache):
                if not cache[0].state:
                    positions_at_start.append((model._position_ids, model._rope_deltas))
                    cache[0].state = [1]
                model._position_ids = "new positions"
                model._rope_deltas = "new deltas"
                return SimpleNamespace(logits=FakeLogits(tokens.values[-1] + 1))

        backend._model = LanguageModel()
        for _ in range(2):
            result = backend.measure([1, 2, 3], 3)
            self.assertEqual(result["generated_token_ids"], [4, 5, 6])
        self.assertEqual(positions_at_start, [(None, None), (None, None)])


if __name__ == "__main__":
    unittest.main()
