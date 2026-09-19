"""CPU-only correctness tests for the nontrivial checkpoint conversion.

These test packed-weight arithmetic and the two normalization conventions;
they do not import MLX, initialize MPS or download any model.
"""

import importlib.util
import unittest

from inference_lab.backends.apple.transformers.backend import TransformersBackend

HAS_TORCH = importlib.util.find_spec("torch") is not None


class CheckpointKeyTests(unittest.TestCase):
    def test_text_names_have_one_transformers_prefix(self):
        cases = {
            "language_model.model.layers.0.mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
            "language_model.lm_head.scales": "lm_head.scales",
            "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
            "model.layers.3.self_attn.q_proj.biases": "model.layers.3.self_attn.q_proj.biases",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(TransformersBackend._text_key(original), expected)

    def test_vision_and_mtp_are_excluded(self):
        for name in ("vision_tower.patch_embed.proj.weight", "model.visual.layers.0.weight",
                     "language_model.mtp.layers.0.weight", "mtp.layers.0.weight"):
            with self.subTest(name=name):
                self.assertIsNone(TransformersBackend._text_key(name))

    def test_cache_quantization_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "quantized KV"):
            TransformersBackend("unused", kv_bits=4)


@unittest.skipUnless(HAS_TORCH, "CPU conversion tests require PyTorch")
class PackedWeightConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_dequantization_preserves_unsigned_high_bits_and_group_offsets(self):
        torch = self.torch
        for bits in (2, 4, 8):
            with self.subTest(bits=bits):
                # More than 1024 rows also tests the bounded-memory chunk edge.
                rows, width, group_size = 1025, 128, 64
                count = 32 // bits
                quantized = torch.arange(rows * width, dtype=torch.int32).reshape(rows, width)
                quantized = ((quantized * 7 + 3) % (1 << bits))
                quantized[:, count - 1::count] = (1 << bits) - 1
                packed = torch.zeros((rows, width // count), dtype=torch.int64)
                for offset in range(count):
                    packed |= quantized[:, offset::count].to(torch.int64) << (bits * offset)
                self.assertTrue((packed > 2**31 - 1).any().item())
                packed = packed.to(torch.uint32)
                scales = torch.tensor([[0.125, 0.25]], dtype=torch.bfloat16).expand(rows, -1)
                biases = torch.tensor([[-2.0, 3.0]], dtype=torch.bfloat16).expand(rows, -1)
                restored = TransformersBackend._dequantize_embedding(
                    packed, scales, biases, bits, group_size, torch)
                expected = (quantized.reshape(rows, -1, group_size).float()
                            * scales.float()[..., None] + biases.float()[..., None])
                self.assertEqual(restored.dtype, torch.bfloat16)
                torch.testing.assert_close(restored, expected.reshape(rows, width).to(torch.bfloat16),
                                           rtol=0, atol=0)

    def test_norm_inverse_preserves_effective_mlx_scale(self):
        torch = self.torch
        values = torch.tensor([0.875, 1.0078125, 1.5, 2.0], dtype=torch.bfloat16)
        for name in ("model.norm.weight", "model.layers.0.input_layernorm.weight",
                     "model.layers.0.post_attention_layernorm.weight",
                     "model.layers.3.self_attn.q_norm.weight", "model.layers.3.self_attn.k_norm.weight"):
            with self.subTest(name=name):
                restored = TransformersBackend._restore_mlx_layout(name, values, sanitized=True)
                self.assertEqual(restored.dtype, torch.float32)
                torch.testing.assert_close(1 + restored, values.float(), rtol=0, atol=0)
                self.assertIs(TransformersBackend._restore_mlx_layout(name, values, sanitized=False), values)

    def test_gated_norm_is_not_zero_centered(self):
        torch = self.torch
        values = torch.tensor([1.0, 1.5], dtype=torch.bfloat16)
        self.assertIs(TransformersBackend._restore_mlx_layout(
            "model.layers.0.linear_attn.norm.weight", values, sanitized=True), values)

    def test_convolution_axes_match_torch_depthwise_layout(self):
        torch = self.torch
        mlx_weight = torch.arange(12).reshape(3, 4, 1)
        restored = TransformersBackend._restore_mlx_layout(
            "model.layers.0.linear_attn.conv1d.weight", mlx_weight, sanitized=True)
        self.assertEqual(tuple(restored.shape), (3, 1, 4))
        torch.testing.assert_close(restored[:, 0, :], mlx_weight[:, :, 0])
        self.assertTrue(restored.is_contiguous())


if __name__ == "__main__":
    unittest.main()
