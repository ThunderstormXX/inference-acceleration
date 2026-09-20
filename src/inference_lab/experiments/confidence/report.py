"""Offline, hash-checked reports for the frozen-policy held-out experiment."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path

from inference_lab.core.io import sha256_file
from inference_lab.visualization.replay_logs import BenchmarkReplayBuilder
from inference_lab.visualization.trace import display_tokens, _enrich_events, parity
from inference_lab.visualization.trajectory import TokenTrajectory
from .runner import RequestGuard, software_environment, validate_budget, verified_json
from .calibration import _classification


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class ConfidenceReport:
    METHODS = ('ar_before','stock_mtp','confidence_collect','confidence_gate','ar_after')
    LABELS = {'ar_before':'baseline','stock_mtp':'mtp','confidence_collect':'MTP + confidence',
              'confidence_gate':'MTP + gate','ar_after':'AR after'}

    def __init__(self, run, policy):
        self.run, self.policy_path = Path(run), Path(policy)

    @staticmethod
    def _rows(path, digest):
        path = Path(path)
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or sha256_file(path) != digest:
            raise ValueError('Sample file hash mismatch')
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _validate_observations(summary, rows, budget):
        config = summary['config']
        if config.get('max_new_tokens') != budget or summary.get('completed_measurements') != len(rows):
            raise ValueError('Summary budget/count differs from saved measurements')
        for row in rows:
            validate_budget(row, budget)
            RequestGuard.validate(row.get('host_observation'), config.get('allow_battery', False), config.get('min_battery', 20))
            prompt = row.get('prompt_token_ids')
            if (not isinstance(prompt, list) or not prompt or any(type(t) is not int or t < 0 for t in prompt)
                    or row.get('prompt_tokens') != len(prompt)
                    or row.get('prompt_token_sha256') != hashlib.sha256(json.dumps(prompt).encode()).hexdigest()):
                raise ValueError('Saved prompt token hash/count mismatch')
        for warmup in summary.get('warmups', []):
            if (warmup.get('generated_tokens') != config.get('warmup_tokens')
                    or warmup.get('requested_tokens') != config.get('warmup_tokens')):
                raise ValueError('Saved warmup budget mismatch')
            RequestGuard.validate(warmup.get('host_observation'), config.get('allow_battery', False), config.get('min_battery', 20))

    @staticmethod
    def _runtime_hashes(metadata):
        # Historical reports compare saved identities, not today's installed files.
        result = {}
        for name, record in metadata.get('upstream_runtime_sources', {}).items():
            if not isinstance(record, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(record.get('sha256', ''))):
                raise ValueError('Invalid saved runtime source hash')
            result[name] = record['sha256']
        return result

    def load(self):
        summary = json.loads((self.run/'summary.json').read_text())
        policy = verified_json(self.policy_path, summary['sources']['policy_sha256'])
        calibration = verified_json(self.policy_path.with_name('calibration.json'), policy['calibration_report_sha256'])
        if summary.get('status') != 'completed' or summary['config'].get('stage') != 'evaluate':
            raise ValueError('Need a complete held-out evaluation')
        if policy.get('status') != 'frozen' or calibration.get('status') != 'completed':
            raise ValueError('Calibration hash/status mismatch')
        for key in ('selected_threshold', 'calibration_source_row_indices', 'expected_max_new_tokens',
                    'source_provenance', 'source_samples_path', 'source_samples_sha256',
                    'source_summary_path', 'source_summary_sha256', 'selection'):
            if key not in policy or calibration.get(key) != policy[key]:
                raise ValueError(f'Frozen policy differs from calibration report: {key}')
        threshold = policy['selected_threshold']
        budget = policy['expected_max_new_tokens']
        if (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1
                or type(budget) is not int or budget < 2 or summary.get('selected_threshold') != threshold):
            raise ValueError('Invalid or inconsistent frozen threshold/budget')
        provenance = policy['source_provenance']
        if provenance.get('policy_sha256') is not None:
            raise ValueError('Calibration unexpectedly depends on a held-out policy')
        for key in ('data_manifest_sha256', 'model_manifest_sha256', 'draft_manifest_sha256', 'code'):
            if key not in provenance or summary['sources'].get(key) != provenance[key]:
                raise ValueError(f'Calibration/evaluation provenance differs: {key}')
        hashes = [provenance[k] for k in ('data_manifest_sha256', 'model_manifest_sha256', 'draft_manifest_sha256')]
        if not isinstance(provenance['code'], dict) or not provenance['code']:
            raise ValueError('Missing measurement code provenance')
        if any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes + list(provenance['code'].values())):
            raise ValueError('Invalid provenance hash')
        collect_summary = verified_json(policy['source_summary_path'], policy['source_summary_sha256'])
        collect_rows = self._rows(policy['source_samples_path'], policy['source_samples_sha256'])
        indices = policy['calibration_source_row_indices']
        if (not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices)
                or collect_summary.get('status') != 'completed' or collect_summary['config'].get('stage') != 'collect'
                or collect_summary.get('sources') != provenance or collect_summary.get('methods') != ['confidence_collect']
                or collect_summary.get('samples_sha256') != policy['source_samples_sha256']
                or collect_summary.get('source_row_indices') != indices
                or [r.get('source_row_index') for r in collect_rows] != indices
                or any(r.get('method') != 'confidence_collect' for r in collect_rows)):
            raise ValueError('Calibration raw coverage/provenance differs from frozen policy')
        self._validate_observations(collect_summary, collect_rows, budget)
        rounds_dataset = policy.get('rounds_dataset')
        if rounds_dataset != calibration.get('rounds_dataset'):
            raise ValueError('Frozen policy round dataset differs from calibration report')
        if rounds_dataset is not None:
            filename = rounds_dataset.get('filename')
            if (not isinstance(filename, str) or Path(filename).name != filename or filename in ('', '.', '..')
                    or rounds_dataset.get('schema_version') != 1
                    or rounds_dataset.get('unit') != 'verified_full_two_proposal_round'
                    or type(rounds_dataset.get('count')) is not int or rounds_dataset['count'] < 1):
                raise ValueError('Invalid frozen calibration round dataset metadata')
            flat_rounds = self._rows(self.policy_path.parent/filename, rounds_dataset.get('sha256'))
            if (len(flat_rounds) != rounds_dataset['count']
                    or any(record.get('source_row_index') not in indices for record in flat_rounds)):
                raise ValueError('Frozen calibration round dataset coverage/count mismatch')
        if software_environment(summary.get('environment')) != software_environment(collect_summary.get('environment')):
            raise ValueError('Calibration/evaluation software environments differ')
        saved_runtime = self._runtime_hashes(collect_summary.get('method_metadata', {}).get('confidence_collect', {}))
        current_runtime = self._runtime_hashes(summary.get('method_metadata', {}).get('confidence_collect', {}))
        if saved_runtime != current_runtime:
            raise ValueError('Calibration/evaluation runtime source hashes differ')
        for metadata in summary.get('method_metadata', {}).values():
            for name, digest in self._runtime_hashes(metadata).items():
                if name in saved_runtime and saved_runtime[name] != digest:
                    raise ValueError('Evaluation methods have inconsistent runtime source hashes')
        gate = summary.get('method_metadata', {}).get('confidence_gate', {})
        if gate.get('confidence_policy') != 'gate' or gate.get('confidence_threshold') != threshold:
            raise ValueError('Actual gate metadata differs from frozen policy')
        rows = self._rows(self.run/'samples.jsonl', summary['samples_sha256'])
        if (tuple(row['method'] for row in rows) != self.METHODS or summary.get('methods') != list(self.METHODS)
                or len({r['source_row_index'] for r in rows}) != 1):
            raise ValueError('Expected exactly one held-out chain across the five prescribed methods')
        index = rows[0]['source_row_index']
        if summary.get('source_row_indices') != [index] or index in indices:
            raise ValueError('Held-out chain leaked into calibration or summary index differs')
        manifest = verified_json(summary['config']['data_manifest'], summary['sources']['data_manifest_sha256'])
        calibration_indices = manifest['splits']['calibration']['source_row_indices']
        heldout_indices = manifest['splits']['heldout']['source_row_indices']
        if (manifest.get('status') != 'completed' or calibration_indices != indices or index not in heldout_indices
                or set(calibration_indices) & set(heldout_indices)):
            raise ValueError('Held-out/calibration dataset split mismatch or leakage')
        self._validate_observations(summary, rows, budget)
        for row in rows:
            if row['prompt_token_ids'] != rows[0]['prompt_token_ids'] or row.get('problem') != rows[0].get('problem'):
                raise ValueError('Method prompts differ')
            if row['generation_trace']['eos_token_ids'] != rows[0]['generation_trace']['eos_token_ids']:
                raise ValueError('Method EOS policies differ')
        return summary, rows, policy, calibration

    @classmethod
    def _trajectory_statistics(cls, lanes):
        paths = {method: TokenTrajectory(lanes[cls.LABELS[method]]) for method in cls.METHODS}
        baseline = paths['ar_before']
        start, end = max(p.prefill for p in paths.values()), min(p.end for p in paths.values())
        if end <= start:
            raise ValueError('No shared decode window across the five methods')
        return {
            'baseline_method': 'ar_before',
            'common_window': {'start_seconds': start, 'end_seconds': end,
                              'definition': 'After all five prefills, until the first of all five final commits'},
            'common_window_comparisons': {name: path.compare(baseline, window=(start, end)) for name, path in paths.items()},
            'pair_window_comparisons': {name: path.compare(baseline) for name, path in paths.items()},
            'interpretation': 'Descriptive observed pacing of sequential requests; equal time window is not repeated-run uncertainty.'}

    def build(self, output):
        summary, rows, policy, calibration = self.load()
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        model=Path(summary['config']['model_path'])
        BenchmarkReplayBuilder._verified_tokenizer_assets(model,summary['sources']['model_manifest_sha256'])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        lanes={}; metrics={}
        baseline=rows[0]
        for row in rows:
            lane=deepcopy(row['generation_trace'])
            lane.update(display_tokens(tokenizer,lane['token_ids']))
            _enrich_events(lane,tokenizer)
            by_round={r['round']:r for r in row.get('confidence_rounds',[])}
            for event in lane['events']:
                confidence=by_round.get(event.get('confidence_round', event.get('round')))
                if confidence:
                    event['confidence']={k:confidence[k] for k in ('p1','p2','confidence_product','decision')}
            lanes[self.LABELS[row['method']]]=lane
            eos=set(lane['eos_token_ids']);first_eos=next((i+1 for i,t in enumerate(lane['token_ids']) if t in eos),None)
            eos_time=next((e['t'] for e in lane['events'] if e['type']=='commit' and first_eos is not None and e['output_count']>=first_eos),None)
            ids=row['generated_token_ids'];reference=baseline['generated_token_ids']
            mismatch=parity(reference, ids)['first_mismatch_index']
            rounds=row.get('confidence_rounds',[])
            metrics[row['method']]={
                'prefill_seconds':row['prefill_seconds'],'prefill_tokens_per_second':row['prompt_tokens']/row['prefill_seconds'],
                'decode_seconds':row['decode_seconds'],'decode_tokens_per_second':row['decode_tokens']/row['decode_seconds'],
                'total_seconds':row['prefill_seconds']+row['decode_seconds'],
                'visible_completion_seconds':lane['events'][-1]['t'],
                'decode_speed_ratio_vs_ar_before':baseline['decode_seconds']/row['decode_seconds'],
                'total_speed_ratio_vs_ar_before':(baseline['prefill_seconds']+baseline['decode_seconds'])/(row['prefill_seconds']+row['decode_seconds']),
                'exact_ids_equal_to_ar_before':ids==reference,'first_mismatch_index_0based':mismatch,
                'generated_tokens':len(ids),'first_eos_output_position_1based':first_eos,'first_eos_observed_seconds':eos_time,
                'peak_memory_gb':row.get('peak_memory_gb'),
                'acceptance':{k:row[k] for k in ('drafted_tokens','accepted_draft_tokens','draft_acceptance_rate','speculative_rounds',
                            'computed_draft_tokens','verified_draft_tokens','gated_out_draft_tokens','fallback_rounds') if k in row},
                'rounds':len(rounds),'fallback_rounds_observed':sum(r['decision']=='fallback' for r in rounds),
                'power_percent_before':row['host_observation']['power_before']['percent'],
                'power_percent_after':row['host_observation']['power_after']['percent'],
                'sleep_seconds':row['host_observation']['sleep_assessment']['sleep_seconds']}
        collect=next(r for r in rows if r['method']=='confidence_collect')
        records=[{'score':r['confidence_product'],'accepted_count':r['accepted_count']} for r in collect['confidence_rounds']
                 if r['decision']=='verify' and len(r['proposal_token_ids'])==2]
        public_calibration={k:v for k,v in calibration.items() if not k.endswith('_path')}
        metadata={'dataset_index':baseline['source_row_index'],'label':f"Held-out row {baseline['source_row_index']} · {baseline['generated_tokens']} tokens · one sequential request per method",
                  'source':f"Saved generation traces; frozen calibration rows {policy['calibration_source_row_indices']}; disjoint held-out row {baseline['source_row_index']}",
                  'sampling':'greedy','enable_thinking':True,'ignore_eos':True,
                  'timing_origin':'each sequential request prefill start; observed times are not interpolated'}
        trace={'schema_version':1,'metadata':metadata,'prompt':baseline['problem'],'prompt_token_ids':baseline['prompt_token_ids'],'lanes':lanes}
        dump(output/'trace.json',trace)
        # A replay-compatible pair for the existing GIF renderer; no inference is repeated.
        for method in ('stock_mtp','confidence_gate'):
            pair={'schema_version':1,'metadata':metadata,'prompt':baseline['problem'],'prompt_token_ids':baseline['prompt_token_ids'],
                  'max_new_tokens':len(lanes['baseline']['token_ids']),'eos_token_ids':lanes['baseline']['eos_token_ids'],
                  'baseline':lanes['baseline'],'mtp':lanes[self.LABELS[method]],
                  'parity':parity(lanes['baseline']['token_ids'],lanes[self.LABELS[method]]['token_ids'])}
            dump(output/f'{method}-replay.json',pair)
        public={'schema_version':1,'selected_threshold':policy['selected_threshold'],'heldout_source_row_index':baseline['source_row_index'],
                'methods':metrics,'calibration':public_calibration,
                'trajectory_statistics':self._trajectory_statistics(lanes),
                'heldout_shadow_classification':_classification(records,policy['selected_threshold']),
                'shadow_caveat':'Acceptance labels from independent always-verify confidence run; not counterfactual timings or labels for unverified gated proposals.',
                'provenance':{'summary_sha256':sha256_file(self.run/'summary.json'),'samples_sha256':summary['samples_sha256'],
                              'policy_sha256':sha256_file(self.policy_path),'calibration_sha256':sha256_file(self.policy_path.with_name('calibration.json')),
                              'data_manifest_sha256':summary['sources']['data_manifest_sha256'],
                              'model_manifest_sha256':summary['sources']['model_manifest_sha256'],
                              'draft_manifest_sha256':summary['sources']['draft_manifest_sha256'],
                              'code':summary['sources']['code']},
                'limits':'Single held-out prompt, single observation per method; endpoint power recorded, sequential order, observer cost included. No confidence interval or dataset-wide timing claim.'}
        dump(output/'experiment.json',public)
        return public

    @staticmethod
    def render(output):
        from inference_lab.visualization.trajectory import TrajectoryReport
        output=Path(output)
        trace=json.loads((output/'trace.json').read_text())
        TrajectoryReport(trace).write(output)
        # Rejection examples with scores from the always-verify confidence lane.
        confidence_trace={'lanes':{'baseline':trace['lanes']['baseline'],'mtp':trace['lanes']['MTP + confidence']}}
        examples=TrajectoryReport(confidence_trace).rejection_examples()
        TrajectoryReport.draw_rejections(examples,output/'rejections.png')
        dump(output/'rejection-examples.json',examples)
        report=json.loads((output/'experiment.json').read_text())
        import matplotlib.pyplot as plt
        c=report['calibration'];fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
        for label,key in [('p1 × p2','product_vs_both_accepted'),('p1','p1_vs_first_accepted'),('p2 | first accepted','p2_vs_second_accepted_given_first')]:
            bins=[b for b in c['score_reliability'][key]['bins'] if b['count']]
            axes[0].plot([b['mean_score'] for b in bins],[b['observed_acceptance_rate'] for b in bins],marker='o',label=label)
        axes[0].plot([0,1],[0,1],ls='--',color='gray',alpha=.6)
        axes[0].set(title='Draft likelihood ≠ target acceptance',xlabel='Mean score in bin',ylabel='Observed acceptance frequency',xlim=(0,1),ylim=(0,1));axes[0].legend(fontsize=9)
        grid=c['threshold_grid_results'];x=[r['threshold'] for r in grid]
        axes[1].plot(x,[r['macro_balanced_accuracy'] for r in grid],label='Balanced accuracy')
        axes[1].plot(x,[r['macro_coverage'] for r in grid],label='Fraction verified')
        axes[1].axvline(c['selected_threshold'],color='#c43c50',ls='--',label=f"Frozen threshold {c['selected_threshold']:.2f}")
        axes[1].set(title='Calibration only · equal weight per chain',xlabel='Threshold on p1 × p2',ylim=(0,1));axes[1].legend(fontsize=9)
        folds=c['leave_one_chain_out']['folds']
        axes[2].bar([str(f['left_out_source_row_index']) for f in folds],[f['selected_threshold'] for f in folds],color='#26a69a')
        axes[2].set(title=f"Threshold stability · train on {len(folds)-1}, validate on 1",xlabel='Left-out calibration chain',ylabel='Selected threshold',ylim=(0,1))
        for ax in axes:ax.grid(alpha=.15)
        fig.suptitle(f"{len(c['calibration_source_row_indices'])} fresh calibration chains × {c['expected_max_new_tokens']} tokens · held-out row {report['heldout_source_row_index']} excluded",fontweight='bold')
        fig.savefig(output/'calibration.png',dpi=170);fig.savefig(output/'calibration.svg');plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path);parser.add_argument('--policy',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args(argv)
    if args.render_only:ConfidenceReport.render(args.output)
    else:
        if not args.run or not args.policy:parser.error('--run and --policy are required to build the report')
        result=ConfidenceReport(args.run,args.policy).build(args.output)
        print(json.dumps({'threshold':result['selected_threshold'],'methods':result['methods']},indent=2))
    return 0
