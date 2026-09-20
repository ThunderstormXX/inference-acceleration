"""Finite, durable calibration and held-out native-MTP experiments."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from inference_lab.core.config import ROOT
from inference_lab.core.host_clock import MacSleepClock
from inference_lab.core.io import environment, resource_snapshot, sha256_file, write_json
from inference_lab.visualization.trace import build_prompt
from inference_lab.visualization.recording import validate_generation_trace
from inference_lab.benchmarking.metrics import validate_measurement


def now():
    return datetime.now(timezone.utc).isoformat()


def append_json(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


class RequestGuard:
    """Detect sleep per request and record endpoint power without changing OS settings."""
    def __init__(self, allow_battery=False, min_battery=20):
        self.clock = MacSleepClock()
        self.allow_battery, self.min_battery = allow_battery, min_battery
    @staticmethod
    def parse_power(raw, allow_battery=False, min_battery=20):
        if not isinstance(raw, str):
            raise ValueError('Missing raw power evidence')
        charge = re.search(r'(\d+)%;', raw)
        source = re.search(r"Now drawing from '(AC Power|Battery Power)'", raw)
        if not charge or not source or not 0 <= int(charge[1]) <= 100:
            raise ValueError('Cannot parse power evidence')
        percent = int(charge[1])
        discharging = bool(re.search(r'\bdischarging\b', raw.lower()))
        allowed = not (discharging and percent < min_battery)
        if not allow_battery:
            allowed = allowed and source[1] == 'AC Power' and not discharging
        return {'raw': raw.strip(), 'percent': percent, 'discharging': discharging, 'allowed': allowed}

    @staticmethod
    def validate(observation, allow_battery=False, min_battery=20):
        if not isinstance(observation, dict):
            raise ValueError('Missing host observation')
        assessed = MacSleepClock.assess(observation.get('clock_before'), observation.get('clock_after'))
        if observation.get('sleep_assessment') != assessed:
            raise ValueError('Saved sleep assessment differs from Mach clock evidence')
        for endpoint in ('power_before', 'power_after'):
            recorded = observation.get(endpoint)
            if not isinstance(recorded, dict):
                raise ValueError('Missing endpoint power evidence')
            parsed = RequestGuard.parse_power(recorded.get('raw'), allow_battery, min_battery)
            if any(type(recorded.get(key)) is not type(value) or recorded.get(key) != value
                   for key, value in parsed.items()):
                raise ValueError('Saved power fields differ from raw endpoint evidence')
            if not parsed['allowed']:
                raise ValueError('Saved endpoint violates power policy')
        if observation.get('valid') is not True or assessed['sleep_detected']:
            raise ValueError('Saved observation contains invalid sleep/power evidence')

    def power(self):
        raw = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        return self.parse_power(raw, self.allow_battery, self.min_battery)
    def start(self):
        power = self.power()
        if not power['allowed']:
            raise RuntimeError('Power guard paused this finite experiment before GPU work')
        return {'power_before': power, 'clock_before': self.clock.snapshot()}
    def finish(self, observation):
        observation['clock_after'] = self.clock.snapshot()
        observation['power_after'] = self.power()
        observation['sleep_assessment'] = self.clock.assess(observation['clock_before'], observation['clock_after'])
        observation['valid'] = not observation['sleep_assessment']['sleep_detected'] and observation['power_after']['allowed']
        return observation


def software_environment(snapshot):
    """Stable software fields only; PID, power, thermal and swap are observations."""
    fields = ('python', 'packages', 'os', 'machine')
    if not isinstance(snapshot, dict) or any(key not in snapshot for key in fields):
        raise ValueError('Missing software environment provenance')
    if not isinstance(snapshot['packages'], dict):
        raise ValueError('Invalid installed package provenance')
    return {key: snapshot[key] for key in fields}


def validate_runtime_sources(metadata):
    for record in metadata.get('upstream_runtime_sources', {}).values():
        if (not isinstance(record, dict) or not isinstance(record.get('path'), str)
                or not isinstance(record.get('sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])
                or sha256_file(Path(record['path'])) != record['sha256']):
            raise ValueError('Installed runtime source hash changed')


def validate_budget(measurement, expected):
    validate_measurement(measurement)
    validate_generation_trace(measurement)
    if measurement['generated_tokens'] != expected:
        raise ValueError('Measurement does not match the exact output budget')


def verified_json(path, digest):
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Missing or invalid calibration file hash')
    path = Path(path)
    if sha256_file(path) != digest:
        raise ValueError(f'Calibration file hash mismatch: {path.name}')
    return json.loads(path.read_text())


@dataclass(frozen=True)
class ConfidenceExperimentConfig:
    stage: str
    output: str
    data_manifest: str = str(ROOT / 'artifacts/data/validation/manifest.json')
    model_path: str = str(ROOT / 'models/qwen3.5-9b-mlx-4bit')
    draft_path: str = str(ROOT / 'models/qwen3.5-9b-mtp-4bit')
    policy_path: str | None = None
    max_new_tokens: int = 2048
    warmup_tokens: int = 2048
    allow_battery: bool = False
    min_battery: int = 20
    def __post_init__(self):
        if self.stage not in ('smoke', 'collect', 'evaluate'):
            raise ValueError('Unknown experiment stage')
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 2:
            raise ValueError('At least two output tokens required')
        if (type(self.warmup_tokens) is not int or self.warmup_tokens < 2
                or type(self.min_battery) is not int or not 0 <= self.min_battery <= 100
                or type(self.allow_battery) is not bool):
            raise ValueError('Invalid warmup or battery threshold')
        if self.stage == 'evaluate' and not self.policy_path:
            raise ValueError('Held-out evaluation requires a frozen calibration policy')


class ConfidenceExperimentRunner:
    def __init__(self, config, resume=False):
        self.config, self.resume = config, resume
        self.output = Path(config.output).resolve()
        self._calibration_software = None
    def inputs(self):
        path = Path(self.config.data_manifest)
        manifest = json.loads(path.read_text())
        split = 'heldout' if self.config.stage == 'evaluate' else 'calibration'
        source = manifest['splits'][split]
        data = Path(source['path'])
        if sha256_file(data) != source['sha256'] or manifest['status'] != 'completed':
            raise ValueError('Validation split hash/status mismatch')
        rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        if [row['source_row_index'] for row in rows] != source['source_row_indices']:
            raise ValueError('Validation split index mismatch')
        if set(manifest['splits']['calibration']['source_row_indices']) & set(manifest['splits']['heldout']['source_row_indices']):
            raise ValueError('Calibration/heldout overlap')
        policy = None
        if self.config.stage == 'evaluate':
            policy = json.loads(Path(self.config.policy_path).read_text())
            if set(policy['calibration_source_row_indices']) != set(manifest['splits']['calibration']['source_row_indices']):
                raise ValueError('Frozen policy was calibrated on a different split')
            if policy.get('status') != 'frozen' or policy.get('expected_max_new_tokens') != self.config.max_new_tokens:
                raise ValueError('Held-out evaluation requires a frozen policy for this output budget')
            if (type(policy.get('selected_threshold')) not in (int, float)
                    or not math.isfinite(policy['selected_threshold']) or not 0 <= policy['selected_threshold'] <= 1):
                raise ValueError('Invalid policy threshold')
            self.validate_policy_evidence(policy)
        if self.config.stage == 'smoke':
            rows = rows[:1]
        return rows, manifest, policy
    def validate_policy_evidence(self, policy):
        provenance = policy.get('source_provenance')
        expected = {
            'data_manifest_sha256': sha256_file(Path(self.config.data_manifest)),
            'model_manifest_sha256': sha256_file(Path(self.config.model_path)/'download-manifest.json'),
            'draft_manifest_sha256': sha256_file(Path(self.config.draft_path)/'download-manifest.json'),
            'policy_sha256': None,
        }
        if not isinstance(provenance, dict) or any(provenance.get(k) != v for k,v in expected.items()):
            raise ValueError('Frozen policy source provenance differs from current data/model/draft')
        if 'code' in provenance and provenance['code'] != self.sources():
            raise ValueError('Frozen policy measurement code differs from current sources')
        summary = None
        for kind in ('summary', 'samples'):
            path_key, hash_key = f'source_{kind}_path', f'source_{kind}_sha256'
            if path_key not in policy and hash_key not in policy:
                continue
            if path_key not in policy or hash_key not in policy:
                raise ValueError('Incomplete calibration file reference')
            path = Path(policy[path_key])
            if sha256_file(path) != policy[hash_key]:
                raise ValueError(f'Calibration {kind} hash mismatch')
            if kind == 'summary':
                summary = verified_json(path, policy[hash_key])
                config = summary.get('config', {})
                if (summary.get('status') != 'completed' or config.get('stage') != 'collect'
                        or config.get('max_new_tokens') != self.config.max_new_tokens
                        or summary.get('methods') != ['confidence_collect']
                        or summary.get('source_row_indices') != policy['calibration_source_row_indices']
                        or summary.get('sources') != provenance):
                    raise ValueError('Calibration summary disagrees with frozen policy')
                self._calibration_software = software_environment(summary.get('environment'))
                for metadata in summary.get('method_metadata', {}).values():
                    validate_runtime_sources(metadata)
            else:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if ([r.get('source_row_index') for r in rows] != policy['calibration_source_row_indices']
                        or any(r.get('method') != 'confidence_collect' for r in rows)):
                    raise ValueError('Calibration sample coverage differs from frozen policy')
                if summary is not None and (summary.get('samples_sha256') != policy[hash_key]
                        or summary.get('completed_measurements') != len(rows)):
                    raise ValueError('Calibration summary samples hash/count mismatch')
                for row in rows:
                    validate_budget(row, self.config.max_new_tokens)
                    if summary is not None:
                        config = summary['config']
                        RequestGuard.validate(row.get('host_observation'), config.get('allow_battery', False), config.get('min_battery', 20))
        if 'calibration_report_sha256' in policy:
            report = verified_json(Path(self.config.policy_path).parent/'calibration.json', policy['calibration_report_sha256'])
            if report.get('status') != 'completed':
                raise ValueError('Calibration report is incomplete')
            for key in ('selected_threshold', 'calibration_source_row_indices', 'expected_max_new_tokens',
                        'source_provenance', 'source_samples_path', 'source_samples_sha256',
                        'source_summary_path', 'source_summary_sha256', 'selection'):
                if key in policy and report.get(key) != policy[key]:
                    raise ValueError(f'Frozen policy differs from calibration report: {key}')
    def sources(self):
        paths = [Path(__file__), ROOT/'src/inference_lab/core/host_clock.py', ROOT/'src/inference_lab/visualization/recording.py',
                 ROOT/'src/inference_lab/visualization/trace.py', ROOT/'requirements-mac.lock']
        paths += sorted((ROOT/'src/inference_lab/backends').rglob('*.py'))
        return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    def backend(self, method, threshold):
        if method.startswith('ar'):
            from inference_lab.backends.apple.mlx_vlm_backend import MLXVLMBackend
            backend = MLXVLMBackend(self.config.model_path, wired_memory=True)
        elif method == 'stock_mtp':
            from inference_lab.backends.apple.speculative.mtp_backend import MTPBackend
            backend = MTPBackend(self.config.model_path, self.config.draft_path, block_size=3)
        else:
            from inference_lab.backends.apple.speculative.confidence_backend import ConfidenceMTPBackend
            policy = {'confidence_collect':'collect', 'confidence_gate':'gate', 'gate_zero':'gate',
                      'gate_one':'gate', 'no_draft':'no_draft'}[method]
            threshold = 0. if method == 'gate_zero' else 1. if method == 'gate_one' else threshold
            backend = ConfidenceMTPBackend(self.config.model_path, self.config.draft_path, threshold=threshold, policy=policy)
        backend.trace_generation = True
        return backend
    def run(self):
        rows, data, policy = self.inputs()
        methods = (['confidence_collect'] if self.config.stage == 'collect' else
                   ['ar_before', 'stock_mtp', 'confidence_collect', 'confidence_gate', 'ar_after'] if self.config.stage == 'evaluate' else
                   ['ar_before', 'stock_mtp', 'confidence_collect', 'gate_zero', 'gate_one', 'no_draft'])
        threshold = policy['selected_threshold'] if policy else 0.
        config = asdict(self.config)
        pinned = {'code': self.sources(), 'data_manifest_sha256': sha256_file(Path(self.config.data_manifest)),
                  'model_manifest_sha256': sha256_file(Path(self.config.model_path)/'download-manifest.json'),
                  'draft_manifest_sha256': sha256_file(Path(self.config.draft_path)/'download-manifest.json'),
                  'policy_sha256': sha256_file(Path(self.config.policy_path)) if policy else None}
        current_environment = environment()
        software = software_environment(current_environment)
        if self._calibration_software is not None and self._calibration_software != software:
            raise ValueError('Calibration and held-out software environments differ')
        self.output.mkdir(parents=True, exist_ok=True)
        summary_path = self.output/'summary.json'
        samples_path = self.output/'samples.jsonl'
        if summary_path.exists():
            if not self.resume:
                raise ValueError('Output exists; use --resume for the unchanged experiment')
            summary = json.loads(summary_path.read_text())
            if summary['config'] != config or summary['sources'] != pinned:
                raise ValueError('Resume protocol or sources changed')
            if software_environment(summary.get('environment')) != software:
                raise ValueError('Resume installed software environment changed')
            for metadata in summary.get('method_metadata', {}).values():
                validate_runtime_sources(metadata)
        else:
            summary = {'schema_version':1, 'created_at':now(), 'config':config, 'sources':pinned,
                       'environment':current_environment, 'methods':methods, 'selected_threshold':threshold,
                       'split_revision':data['source_revision'], 'source_row_indices':[r['source_row_index'] for r in rows],
                       'method_metadata':{}, 'warmups':[], 'status':'running',
                       'methodology':'Greedy batch1, fresh cache; instrumented; fixed output budget ignoring EOS; explicit battery policy; selection frozen before heldout.'}
        saved = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()] if samples_path.exists() else []
        expected = {(method,row['source_row_index']) for method in methods for row in rows}
        completed = {(row['method'],row['source_row_index']) for row in saved}
        if saved and summary.get('samples_sha256') != sha256_file(samples_path):
            raise ValueError('Saved sample file does not match its last durable summary hash')
        if len(saved) != len(completed) or not completed.issubset(expected):
            raise ValueError('Duplicate or unexpected saved measurements')
        for row in saved:
            validate_budget(row, self.config.max_new_tokens)
            RequestGuard.validate(row.get('host_observation'), self.config.allow_battery, self.config.min_battery)
        for warmup in summary.get('warmups', []):
            if (warmup.get('generated_tokens') != self.config.warmup_tokens
                    or warmup.get('requested_tokens') != self.config.warmup_tokens):
                raise ValueError('Saved warmup does not match the exact output budget')
            RequestGuard.validate(warmup.get('host_observation'), self.config.allow_battery, self.config.min_battery)
        guard = RequestGuard(self.config.allow_battery, self.config.min_battery)
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            backend = None
            try:
                summary.update(status='running', resumed_at=now())
                write_json(summary_path,summary)
                for method in methods:
                    todo = [r for r in rows if (method,r['source_row_index']) not in completed]
                    if not todo: continue
                    if self.sources() != pinned['code']:
                        raise ValueError('Measurement code changed during experiment')
                    guard.start()
                    backend = self.backend(method,threshold)
                    backend.load()
                    metadata = backend.metadata()
                    validate_runtime_sources(metadata)
                    previous_metadata = summary['method_metadata'].get(method)
                    if previous_metadata is not None and previous_metadata != metadata:
                        raise ValueError('Method runtime metadata changed; refusing to overwrite provenance')
                    summary['method_metadata'][method] = metadata
                    prompt = build_prompt(backend.tokenizer, todo[0]['problem'], enable_thinking=True)
                    observation = guard.start()
                    print(f'{method}: warmup {self.config.warmup_tokens} tokens',flush=True)
                    warmup_measurement = backend.measure(prompt,self.config.warmup_tokens)
                    guard.finish(observation)
                    warmup_record = {'method':method,'host_observation':observation,
                        'requested_tokens':self.config.warmup_tokens,'generated_tokens':warmup_measurement.get('generated_tokens'),
                        'source_row_index':todo[0]['source_row_index']}
                    try:
                        if not observation['valid']:
                            raise RuntimeError('Warmup crossed sleep/power guard; no measurements accepted')
                        validate_budget(warmup_measurement, self.config.warmup_tokens)
                        RequestGuard.validate(observation, self.config.allow_battery, self.config.min_battery)
                    except Exception:
                        summary.setdefault('invalid_warmups', []).append(warmup_record)
                        raise
                    summary['warmups'].append(warmup_record)
                    write_json(summary_path,summary)
                    for source in todo:
                        index = source['source_row_index']
                        prompt = build_prompt(backend.tokenizer,source['problem'],enable_thinking=True)
                        observation = guard.start()
                        measurement = backend.measure(prompt,self.config.max_new_tokens)
                        guard.finish(observation)
                        measurement.update(source_row_index=index,index=index,method=method,problem=source['problem'],
                                           prompt_token_ids=prompt, prompt_token_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                                           host_observation=observation)
                        validate_budget(measurement, self.config.max_new_tokens)
                        if not observation['valid']:
                            append_json(self.output/'invalid-samples.jsonl',measurement)
                            raise RuntimeError('Request crossed sleep/power guard; retained separately as invalid')
                        RequestGuard.validate(observation, self.config.allow_battery, self.config.min_battery)
                        if self.sources() != pinned['code']:
                            raise ValueError('Measurement code changed during request')
                        append_json(samples_path,measurement)
                        saved.append(measurement)
                        completed.add((method,index))
                        summary['completed_measurements'] = len(saved)
                        summary['samples_sha256'] = sha256_file(samples_path)
                        write_json(summary_path,summary)
                        print(f'{method} row={index} tokens={measurement["generated_tokens"]} decode={measurement["decode_tokens"]/measurement["decode_seconds"]:.2f} tok/s',flush=True)
                    mx = backend._mx
                    del backend
                    backend = None
                    gc.collect(); mx.synchronize(); mx.clear_cache()
                by_index = {}
                for row in saved:
                    by_index.setdefault(row['source_row_index'],{})[row['method']] = row['generated_token_ids']
                summary['parity'] = {str(index):{'equal':None if len(lanes)<2 else all(ids==next(iter(lanes.values())) for ids in lanes.values()),
                                             'methods':list(lanes)} for index,lanes in by_index.items()}
                summary.update(status='completed',finished_at=now(),completed_measurements=len(saved),
                               samples_sha256=sha256_file(samples_path),resources_after=resource_snapshot())
            except BaseException as exc:
                summary.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}',resources_after=resource_snapshot())
                raise
            finally:
                write_json(summary_path,summary)
                if backend is not None:
                    mx = backend._mx
                    del backend
                    gc.collect(); mx.synchronize(); mx.clear_cache()
        print(f'Completed: {self.output}',flush=True)
        return self.output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['smoke','collect','evaluate'])
    p.add_argument('--output',required=True)
    p.add_argument('--data-manifest',default=ConfidenceExperimentConfig.data_manifest)
    p.add_argument('--policy-path')
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--warmup-tokens',type=int,default=2048)
    p.add_argument('--allow-battery',action='store_true')
    p.add_argument('--min-battery',type=int,default=20)
    p.add_argument('--resume',action='store_true')
    args=vars(p.parse_args(argv));resume=args.pop('resume')
    ConfidenceExperimentRunner(ConfidenceExperimentConfig(**args),resume).run()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
