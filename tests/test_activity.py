"""Mock-only activity lifecycle tests; never change the host's activity state."""

import json
from types import SimpleNamespace

import pytest

from inference_lab.core.activity import (
    USER_INITIATED_OPTIONS,
    UserInitiatedActivity,
    _FoundationActivityAPI,
)
from inference_lab.core.config import BenchmarkConfig


class FakeAPI:
    def __init__(self):
        self.events = []
        self.handle = (object(), object())

    def begin(self, options, reason):
        self.events.append(("begin", options, reason))
        return self.handle

    def end(self, handle):
        assert handle is self.handle
        self.events.append(("end",))


def test_disabled_context_never_loads_foundation():
    def forbidden_factory():
        raise AssertionError("Foundation must not load in the baseline")
    with UserInitiatedActivity(_api_factory=forbidden_factory) as activity:
        assert activity.metadata()["active"] is False
    assert activity.metadata()["started"] is False
    assert activity.metadata()["options"] == 0


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_token_is_held_for_context_and_ended_once_after_exception(error):
    api = FakeAPI()
    activity = UserInitiatedActivity(True, "Finite test", _api_factory=lambda: api)

    def work():
        with activity:
            assert activity.metadata()["active"] is True
            assert api.events == [("begin", 0x00FFFFFF, "Finite test")]
            if error:
                raise error("interrupted work")

    if error:
        with pytest.raises(error):
            work()
    else:
        work()
    activity.close()
    assert api.events == [("begin", 0x00FFFFFF, "Finite test"), ("end",)]
    assert activity.metadata()["active"] is False
    assert activity.metadata()["ended"] is True
    assert USER_INITIATED_OPTIONS & (1 << 40) == 0
    assert USER_INITIATED_OPTIONS & 0xFF00000000 == 0


def test_single_use_prevents_overwriting_a_retained_token():
    api = FakeAPI()
    activity = UserInitiatedActivity(True, _api_factory=lambda: api)
    with activity:
        with pytest.raises(RuntimeError, match="more than once"):
            activity.__enter__()
    assert [event[0] for event in api.events] == ["begin", "end"]


def fake_objective_c_bridge(*, missing_token=False, end_fails=False):
    # Exercise the real ownership logic with Objective-C message calls replaced.
    api = _FoundationActivityAPI.__new__(_FoundationActivityAPI)
    events = []
    api._pool = lambda: "pool"
    api._class = lambda name: name
    api._selector = lambda name: name

    def object_message(receiver, selector):
        events.append((receiver, selector))
        return "process" if selector == "processInfo" else receiver

    def begin(receiver, selector, options, reason):
        events.append((receiver, selector, options, reason))
        return None if missing_token else "token"

    def void_object(receiver, selector, token):
        events.append((receiver, selector, token))
        if end_fails:
            raise RuntimeError("mock end error")

    api._object = object_message
    api._string = lambda *args: "reason"
    api._begin = begin
    api._void = lambda receiver, selector: events.append((receiver, selector))
    api._void_object = void_object
    return api, events


def test_native_token_is_retained_before_autorelease_pool_drains():
    api, events = fake_objective_c_bridge()
    handle = api.begin(USER_INITIATED_OPTIONS, "Test")
    assert handle == ("process", "token")
    assert events.index(("process", "retain")) < events.index(("pool", "drain"))
    assert events.index(("token", "retain")) < events.index(("pool", "drain"))
    api.end(handle)
    assert events[-3:] == [
        ("process", "endActivity:", "token"), ("token", "release"), ("process", "release")
    ]


def test_failed_begin_releases_process_and_drains_pool():
    api, events = fake_objective_c_bridge(missing_token=True)
    with pytest.raises(RuntimeError, match="no activity token"):
        api.begin(USER_INITIATED_OPTIONS, "Test")
    assert events[-2:] == [("process", "release"), ("pool", "drain")]


def test_end_failure_still_releases_both_native_references():
    api, events = fake_objective_c_bridge(end_fails=True)
    with pytest.raises(RuntimeError, match="mock end error"):
        api.end(("process", "token"))
    assert events[-2:] == [("token", "release"), ("process", "release")]


@pytest.mark.parametrize("configured, arguments, expected", [
    (False, [], False), (False, ["--user-initiated"], True),
    (True, ["--no-user-initiated"], False),
])
def test_cli_activity_flag_is_explicit_and_configurable(monkeypatch, tmp_path, configured, arguments, expected):
    from inference_lab.benchmarking import cli
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"user_initiated": configured}))
    seen = []
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda config: SimpleNamespace(run=lambda: seen.append(config)))
    monkeypatch.setattr("sys.argv", ["mlx.py", "--config", str(config_file), *arguments])
    cli.main("mlx")
    assert seen[0].user_initiated is expected


def test_default_configuration_does_not_enable_activity():
    assert BenchmarkConfig("mlx").user_initiated is False
    with pytest.raises(ValueError, match="boolean"):
        BenchmarkConfig("mlx", user_initiated="true")
