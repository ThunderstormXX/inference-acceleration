"""Optional raw generation events embedded in durable benchmark measurements."""
from __future__ import annotations

import math
from time import perf_counter

from .trace import EventRecorder


def recording_policy(speculative=False):
    return {
        "enabled": True,
        "version": 1,
        "clock": "perf_counter; origin is the backend prefill start",
        "observer_overhead_included": True,
        "baseline": "commit at existing token.item() reads; existing async pipeline; no extra GPU synchronization",
        "speculative": (
            "proposal IDs read to host before verification; target IDs observed after stock acceptance walk; "
            "commit marks native host cache enqueue, not GPU cache completion; final backend cache eval/sync included"
            if speculative else None
        ),
        "display_decoding": "offline; no Unicode prefix enrichment in measured phases",
        "comparability": "instrumented timings; compare to runs with the same recording policy",
    }


class GenerationRecorder(EventRecorder):
    def __init__(self, backend, clock=perf_counter, *, speculative=False):
        super().__init__(clock)
        self.backend = backend
        self.speculative = speculative

    def start_at(self, started):
        self.started = started

    def finish_measurement(self, measurement):
        eos = getattr(self.backend.tokenizer, "eos_token_id", None)
        eos_ids = [eos] if type(eos) is int else list(eos or [])
        config = getattr(self.backend, "_model_config", {})
        for candidate in (config.get("eos_token_id"), config.get("text_config", {}).get("eos_token_id")):
            eos_ids.extend([candidate] if type(candidate) is int else list(candidate or []))
        return {
            "schema_version": 1, "token_ids": list(self.token_ids), "events": self.events,
            "prefill_seconds": measurement["prefill_seconds"],
            "decode_seconds": measurement["decode_seconds"],
            "total_seconds": measurement["prefill_seconds"] + measurement["decode_seconds"],
            "eos_token_ids": sorted(set(eos_ids)), "instrumentation": recording_policy(self.speculative),
        }


def validate_generation_trace(measurement):
    """Fail closed before writing a row or resuming from saved trace evidence."""
    lane = measurement.get("generation_trace")
    if not isinstance(lane, dict) or lane.get("schema_version") != 1:
        raise ValueError("Missing or unsupported generation_trace")

    def seconds(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid generation trace time: {name}")
        return value

    def tokens(value, name, *, nonempty=False):
        if (not isinstance(value, list) or (nonempty and not value)
                or any(type(token) is not int or token < 0 for token in value)):
            raise ValueError(f"Invalid generation trace token IDs: {name}")
        return value

    generated = tokens(measurement.get("generated_token_ids"), "measurement", nonempty=True)
    if tokens(lane.get("token_ids"), "lane") != generated:
        raise ValueError("Generation trace token IDs disagree with measurement")
    if measurement.get("generated_tokens") != len(generated):
        raise ValueError("Generation trace length differs from generated_tokens")
    tokens(lane.get("eos_token_ids"), "eos")
    if not isinstance(lane.get("instrumentation"), dict) or lane["instrumentation"].get("enabled") is not True:
        raise ValueError("Generation trace must disclose its instrumentation")
    prefill = seconds(lane.get("prefill_seconds"), "prefill")
    decode = seconds(lane.get("decode_seconds"), "decode")
    total = seconds(lane.get("total_seconds"), "total")
    if not math.isclose(total, prefill + decode, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Generation trace total does not equal phase durations")
    for phase in ("prefill_seconds", "decode_seconds"):
        if not math.isclose(lane[phase], seconds(measurement.get(phase), phase), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Generation trace phase time differs from measurement")
    events = lane.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Generation trace requires events")
    committed, previous, pending = [], 0.0, None
    drafts = accepted_total = rounds = 0
    first_commit = None
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid generation trace event")
        current = seconds(event.get("t"), "event")
        if current < previous or current > total + 1e-9:
            raise ValueError("Generation trace events are not chronological within request duration")
        previous = current
        ids = tokens(event.get("token_ids"), "event", nonempty=True)
        if event.get("type") == "draft":
            if pending is not None or not committed or event.get("output_count") != len(committed):
                raise ValueError("Generation draft has invalid committed-prefix count/order")
            pending = event
            continue
        if event.get("type") != "commit":
            raise ValueError("Unknown generation trace event type")
        if first_commit is None:
            first_commit = current
            if len(ids) != 1 or current > prefill + 1e-9:
                raise ValueError("Generation prefill must commit exactly the first token")
        elif current < prefill:
            raise ValueError("Generation decode commit precedes the prefill boundary")
        if pending is not None:
            proposed = pending["token_ids"]
            accepted = event.get("accepted_count")
            if (type(accepted) is not int or not 0 <= accepted <= len(proposed)
                    or event.get("draft_count") != len(proposed)
                    or event.get("round") != pending.get("round")
                    or event.get("proposed_token_ids") != proposed
                    or event.get("rejected_token_ids") != proposed[accepted:]):
                raise ValueError("Generation draft/verification counters disagree")
            if ids[:min(accepted, len(ids))] != proposed[:min(accepted, len(ids))]:
                raise ValueError("Generation committed draft prefix differs from proposals")
            verified = seconds(event.get("verification_completed_t"), "verification")
            enqueued = seconds(event.get("cache_commit_enqueued_t"), "cache enqueue")
            if not pending["t"] <= verified <= enqueued <= current:
                raise ValueError("Generation verification/commit times are out of order")
            drafts += len(proposed)
            accepted_total += accepted
            rounds += 1
            pending = None
        elif event.get("round", 0) != 0:
            raise ValueError("Generation speculative commit is missing its draft")
        committed.extend(ids)
        if type(event.get("output_count")) is not int or event["output_count"] != len(committed):
            raise ValueError("Generation commit output_count disagrees with tokens")
    if pending is not None or committed != generated:
        raise ValueError("Generation commit token IDs disagree with completed measurement")
    for name, value in (("drafted_tokens", drafts), ("accepted_draft_tokens", accepted_total), ("speculative_rounds", rounds)):
        if name in measurement and measurement[name] != value:
            raise ValueError(f"Generation trace counter differs from measurement: {name}")
