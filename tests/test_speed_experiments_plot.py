"""A comparative plot must reject diagnostic measurements before rendering."""
import json
from statistics import mean, stdev

import pytest

from inference_lab.optimizations.plot import load_panel


KEYS = (("ar", "AR"), ("fast", "Fast"))


def save_fixture(directory, *, last_ar=10.2, stale_mean=False, stale_sd=False):
    directory.mkdir()
    runs = []
    for name, speed, repeat, phase in (
        ("ar", 10, -1, "reference"), ("ar", 10, 0, "measurement"),
        ("fast", 20, 0, "measurement"), ("ar", 11, 1, "measurement"),
        ("fast", 22, 1, "measurement"), ("ar", last_ar, 2, "bracket"),
    ):
        runs.append({"variant": name, "prompt_index": 0, "repeat": repeat, "phase": phase,
                     "measurement": {"decode_tokens": 4, "decode_seconds": 4 / speed,
                                     "generated_token_ids": [1, 2, 3, 4, 5]},
                     "matches_baseline": True, "sleep": {"sleep_detected": False}})
    summary = {"status": "completed", "all_token_ids_match": True, "all_runs_awake": True,
               "by_variant": {}}
    for name, rates in (("ar", [10, 11]), ("fast", [20, 22])):
        summary["by_variant"][name] = {"tok_s": {"n": 2, "mean": mean(rates), "stdev": stdev(rates)}}
    if stale_mean:
        summary["by_variant"]["fast"]["tok_s"]["mean"] = 999
    if stale_sd:
        summary["by_variant"]["fast"]["tok_s"]["stdev"] = 999
    raw = {"status": "completed", "runs": runs, "errors": [],
           "protocol": {"variants": ["ar", "fast"], "repeats": 2, "prompts": [{}], "tokens": 5}}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "raw.json").write_text(json.dumps(raw))


def test_valid_panel_keeps_summary_numbers_and_protocol(tmp_path):
    save_fixture(tmp_path / "valid")
    panel = load_panel(tmp_path / "valid" / "summary.json", KEYS, section="by_variant")
    assert panel["rows"][1]["mean"] == 21
    assert panel["rows"][1]["sd"] == pytest.approx(2 ** .5)
    assert panel["rows"][1]["runs"] == 2
    assert panel["protocol"]["repeats"] == 2


@pytest.mark.parametrize("changes,reason", [
    ({"last_ar": 8}, "AR bracket drift -20.0% exceeds 5%"),
    ({"last_ar": 12}, "AR bracket drift +20.0% exceeds 5%"),
    ({"stale_mean": True}, "summary/raw statistics differ"),
    ({"stale_sd": True}, "Summary/raw sample SD differs"),
])
def test_plot_refuses_drift_and_source_warnings(tmp_path, changes, reason):
    save_fixture(tmp_path / "diagnostic", **changes)
    with pytest.raises(ValueError) as error:
        load_panel(tmp_path / "diagnostic", KEYS, section="by_variant")
    assert reason in str(error.value)
