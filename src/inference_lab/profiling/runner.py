"""Bounded experiment runner. All raw timings and token IDs are retained."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

from ..backends.apple.mlx_backend import MLXBackend
from ..backends.apple.speculative.dflash_backend import DFlashBackend
from .dflash import DFlashProfiler, plain_target

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS = ROOT / 'configs/profiling/dflash-prompts.json'


def condition_snapshot():
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'machine': platform.platform()}
    for key, command in [('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm'])]:
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=10)
            result[key] = {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = {'error': str(exc)}
    return result


def summarize(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else None,
            'min': min(values), 'max': max(values)} if values else None


def compact_summary(report):
    rows = report['runs']
    result = {'runs': [], 'by_mode': {}, 'phase_by_block': {}, 'all_token_ids_match': all(
        row.get('matches_baseline', False) for row in rows)}
    for row in rows:
        m = row['measurement']
        item = {k: row[k] for k in ('mode', 'block_size', 'prompt_index', 'repeat', 'matches_baseline')}
        item.update({'decode_ms': m['decode_seconds'] * 1000, 'decode_tokens': m['decode_tokens'],
                     'ms_per_token': m['decode_seconds'] * 1000 / m['decode_tokens'],
                     'tok_s': m['decode_tokens'] / m['decode_seconds']})
        if 'speculative_rounds' in m:
            item.update({'rounds': m['speculative_rounds'], 'acceptance': m['draft_acceptance_rate'],
                         'useful_tokens_per_round': m['decode_tokens'] / m['speculative_rounds']})
        result['runs'].append(item)
    for mode in sorted({(r['mode'], r['block_size']) for r in rows}, key=str):
        selected = [r for r in result['runs'] if (r['mode'], r['block_size']) == mode]
        result['by_mode'][f'{mode[0]}_k{mode[1]}'] = {
            'decode_ms': summarize([r['decode_ms'] for r in selected]),
            'ms_per_token': summarize([r['ms_per_token'] for r in selected]),
            'tok_s': summarize([r['tok_s'] for r in selected]),
        }
    for k in sorted({r['block_size'] for r in rows if r['mode'] == 'phases'}):
        selected = [r for r in rows if r['mode'] == 'phases' and r['block_size'] == k]
        phases = [e for r in selected for e in r['profile']['spans'] if e['metadata'].get('kind') == 'phase']
        result['phase_by_block'][str(k)] = {
            category: {name: summarize([e['inclusive_ms'] for e in phases if e['name'] == name
                                 and (category == 'all' or e['metadata']['first_round'] == (category == 'first'))])
                       for name in ('draft', 'verify', 'acceptance_commit', 'rollback')}
            for category in ('all', 'first', 'steady')}
        result['phase_by_block'][str(k)]['unattributed_ms_per_run'] = summarize([
            r['profile']['unattributed_decode_ms'] for r in selected])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument('--prompt-count', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--detail-rounds', type=int, nargs='+', default=[1, 10, 20])
    parser.add_argument('--probe-repeats', type=int, default=5)
    parser.add_argument('--skip-probe', action='store_true')
    parser.add_argument('--skip-detail', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.prompt_count <= 5 or not 16 <= args.tokens <= 512 or not 1 <= args.repeats <= 5:
        parser.error('Bounded diagnostic: 1..5 prompts, 16..512 tokens, 1..5 repeats')
    if any(k not in (2, 3, 4, 5) for k in args.block_sizes):
        parser.error('block sizes must be 2..5')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'artifacts/profiling' / f'{stamp}-dflash'
    output.mkdir(parents=True, exist_ok=False)
    prompt_bytes = args.prompts.read_bytes()
    prompts = json.loads(prompt_bytes)[:args.prompt_count]
    report = {'schema_version': 1, 'started': condition_snapshot(), 'runs': [],
              'protocol': {'prompts_sha256': sha256(prompt_bytes).hexdigest(), 'prompts_source': str(args.prompts),
                           'prompts': prompts, 'tokens': args.tokens, 'repeats': args.repeats,
                           'block_sizes': args.block_sizes, 'detail_rounds': args.detail_rounds,
                           'notes': ['All timings are synchronized wall time, not GPU kernel timestamps.',
                                     'First generated token belongs to prefill, decode has N-1 tokens.',
                                     'AR has no hidden feature hooks; draft weights remain resident for paired runs.',
                                     'Phases serialize draft/verify/rollback; detail adds per-module barriers.',
                                     'Fine profiles are diagnostic and not production throughput.',
                                     'Only fixed-budget greedy sampling; EOS ignored; output text decoding excluded.']}}

    def save():
        temporary = output / 'raw.json.gz.tmp'
        with gzip.open(temporary, 'wt', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False)
        temporary.replace(output / 'raw.json.gz')
        (output / 'summary.json').write_text(json.dumps(compact_summary(report), indent=2, ensure_ascii=False) + '\n')

    backend = DFlashBackend(str(ROOT / 'models/qwen3.5-9b-mlx-4bit'),
                            str(ROOT / 'models/qwen3.5-9b-dflash'), block_size=args.block_sizes[0], draft_bits=4).load()
    report['backend'] = backend.metadata()
    report['upstream_source_sha256'] = backend._source_sha256
    print(f'Loaded target + draft; results: {output}', flush=True)
    baselines = {}

    def run(mode, pindex, repeat, k):
        backend.block_size = k
        prompt = prompts[pindex]['prompt_tokens']
        started = perf_counter()
        print(f'RUN {mode} K={k} prompt={pindex} repeat={repeat}', flush=True)
        if mode == 'ar':
            with plain_target(backend):
                measurement = MLXBackend.measure(backend, prompt, args.tokens)
            row = {'measurement': measurement}
            if pindex not in baselines:
                baselines[pindex] = measurement['generated_token_ids']
        elif mode == 'stock':
            row = {'measurement': backend.measure(prompt, args.tokens)}
        else:
            row = DFlashProfiler(backend, detail=mode, detail_rounds=args.detail_rounds).measure(prompt, args.tokens)
        ids = row['measurement']['generated_token_ids']
        row.update({'mode': mode, 'block_size': k if mode != 'ar' else 1,
                    'prompt_index': pindex, 'repeat': repeat, 'wall_seconds': perf_counter() - started,
                    'matches_baseline': ids == baselines[pindex]})
        if not row['matches_baseline']:
            row['first_mismatch'] = next((i for i, (a, b) in enumerate(zip(ids, baselines[pindex])) if a != b), None)
        report['runs'].append(row)
        save()
        print(f"DONE {row['measurement']['decode_seconds']*1000:.1f} ms decode; parity={row['matches_baseline']}", flush=True)

    # Warm compilation and both paths. These requests are excluded from results.
    with plain_target(backend):
        MLXBackend.measure(backend, prompts[0]['prompt_tokens'], 16)
    for k in args.block_sizes:
        backend.block_size = k
        backend.measure(prompts[0]['prompt_tokens'], 16)
    report['after_warmup'] = condition_snapshot()
    # Alternate ordering in the second repetition to expose drift.
    for repeat in range(args.repeats):
        for pindex in range(len(prompts)):
            if repeat % 2 == 0:
                run('ar', pindex, repeat, args.block_sizes[0])
            for k in (args.block_sizes if repeat % 2 == 0 else list(reversed(args.block_sizes))):
                run('stock', pindex, repeat, k)
            if repeat % 2:
                run('ar', pindex, repeat, args.block_sizes[0])
    for pindex in range(len(prompts)):
        for k in args.block_sizes:
            run('phases', pindex, 0, k)
    if not args.skip_detail:
        for k in args.block_sizes:
            for mode in ('layers', 'operators'):
                run(mode, 0, 0, k)
    if not args.skip_probe:
        from .batching import run_batching_probe
        with backend._upstream.wired_limit(backend._model, [backend._upstream.generation_stream]):
            report['batching_probe'] = run_batching_probe(
                backend, prompts[0]['prompt_tokens'], baselines[0], repeats=args.probe_repeats,
                prefix_generated=min(32, args.tokens // 2))
        save()
    # Bracket the heavily instrumented part with a final normal pair.
    run('ar', 0, args.repeats, args.block_sizes[0])
    for k in args.block_sizes:
        run('stock', 0, args.repeats, k)
    report['finished'] = condition_snapshot()
    save()
    print(f'COMPLETE {output}', flush=True)
    return 0 if compact_summary(report)['all_token_ids_match'] else 2
