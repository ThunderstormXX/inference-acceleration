"""Global dataset slicing must remain comparable across independent processes."""
import json

import pytest

from inference_lab.core.config import BenchmarkConfig
from inference_lab.data.prompts import PromptDataset


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [100 + int(messages[0]["content"].split("\n")[0])]


def test_slice_preserves_global_indices_and_selects_matching_rows(tmp_path):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("\n".join(json.dumps({"problem": str(i), "response": "teacher"}) for i in range(100)))
    config = BenchmarkConfig("mlx-vlm", dataset_path=str(dataset), start_index=95, count=5)
    samples = PromptDataset(config, Tokenizer()).load()
    assert [sample.index for sample in samples] == [95, 96, 97, 98, 99]
    assert [sample.problem for sample in samples] == ["95", "96", "97", "98", "99"]
    assert [sample.prompt_tokens for sample in samples] == [[195], [196], [197], [198], [199]]


@pytest.mark.parametrize("start", [-1, True, 0.5, "0"])
def test_start_index_requires_nonnegative_integer(start):
    with pytest.raises(ValueError, match="start_index"):
        BenchmarkConfig("mlx-vlm", count=1, start_index=start)


def test_slice_cannot_exceed_pinned_hundred_rows():
    with pytest.raises(ValueError, match="start_index.*count"):
        BenchmarkConfig("mlx-vlm", count=2, start_index=99)
    assert BenchmarkConfig("mlx-vlm", count=1, start_index=99).start_index == 99


def test_short_dataset_fails_even_when_it_contains_count_rows(tmp_path):
    dataset = tmp_path / "short.jsonl"
    dataset.write_text("\n".join(json.dumps({"problem": str(i)}) for i in range(5)))
    with pytest.raises(ValueError, match="Need 6 rows"):
        PromptDataset(BenchmarkConfig("mlx-vlm", count=2, start_index=4, dataset_path=str(dataset)), Tokenizer()).load()


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_trace_generation_requires_boolean(value):
    with pytest.raises(ValueError, match="trace_generation"):
        BenchmarkConfig("mlx-vlm", trace_generation=value)
