"""CPU-only checks for report provenance, observed-time comparisons, and replay."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from inference_lab.core.host_clock import MacSleepClock
from inference_lab.core.io import sha256_file
from inference_lab.experiments.confidence.report import ConfidenceReport
from inference_lab.experiments.confidence.runner import RequestGuard
from inference_lab.visualization.replay_logs import BenchmarkReplayBuilder


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def observation(sleep=0):
    before = dict(schema_version=1, absolute_ticks=100, continuous_ticks=200,
                  timebase_numer=1_000_000_000, timebase_denom=1, sampling_span_ticks=0)
    after = {**before, 'absolute_ticks':102, 'continuous_ticks':202 + sleep}
    power = RequestGuard.parse_power("Now drawing from 'AC Power'\n -InternalBattery-0 90%; charging;")
    return dict(clock_before=before, clock_after=after, sleep_assessment=MacSleepClock.assess(before, after),
                power_before=deepcopy(power), power_after=deepcopy(power), valid=True)


def row(method, index=20, prefill=.5, end=3., speculative=False):
    ids = [100, 101, 102, 103]
    events = [dict(type='commit', t=prefill, token_ids=ids[:1], output_count=1, round=0)]
    rounds = []
    if speculative:
        events += [dict(type='draft', t=1., token_ids=ids[1:3], output_count=1, round=1),
                   dict(type='commit', t=end, token_ids=ids[1:], output_count=4, round=1,
                        accepted_count=2, draft_count=2, proposed_token_ids=ids[1:3], rejected_token_ids=[],
                        verification_completed_t=end-.1, cache_commit_enqueued_t=end-.05)]
        rounds = [dict(round=1, output_count_before=1, proposal_token_ids=ids[1:3],
                       proposal_probabilities=[.8,.9], proposal_logprobs=[math.log(.8),math.log(.9)],
                       p1=.8, p2=.9, confidence_product=.72, decision='verify', accepted_count=2,
                       verified_draft_count=2, emitted_token_ids=ids[1:])]
    else:
        times = [1.,2.,end] if method=='ar_before' else [1.3,2.3,end]
        events += [dict(type='commit', t=t, token_ids=[token], output_count=i+2, round=0)
                   for i,(t,token) in enumerate(zip(times,ids[1:]))]
        if method=='confidence_gate':
            for i,event in enumerate(events[1:],1):
                event['confidence_round']=i
                rounds.append(dict(round=i, output_count_before=i, proposal_token_ids=[999,998],
                                   p1=.1, p2=.2, confidence_product=.02, accepted_count=None,
                                   decision='fallback', verified_draft_count=0, emitted_token_ids=event['token_ids']))
    prompt = [1,2,3]
    result = dict(method=method, source_row_index=index, problem=f'Problem {index}', prompt_tokens=3,
                  prompt_token_ids=prompt, prompt_token_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                  generated_tokens=4, decode_tokens=3, generated_token_ids=ids,
                  prefill_seconds=prefill, decode_seconds=end+.2-prefill, host_observation=observation(),
                  generation_trace=dict(schema_version=1, token_ids=ids, eos_token_ids=[103], events=events,
                      prefill_seconds=prefill, decode_seconds=end+.2-prefill, total_seconds=end+.2,
                      instrumentation={'enabled':True}))
    if rounds: result['confidence_rounds']=rounds
    if speculative:
        result.update(drafted_tokens=2, accepted_draft_tokens=2, draft_acceptance_rate=1., speculative_rounds=1)
    return result


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    evaluation=tmp_path/'evaluation';evaluation.mkdir()
    collect=tmp_path/'collect';collect.mkdir()
    policy_dir=tmp_path/'policy';policy_dir.mkdir()
    manifest=tmp_path/'data-manifest.json'
    write(manifest, {'status':'completed', 'splits':{
        'calibration':{'source_row_indices':[10,11]}, 'heldout':{'source_row_indices':[20]}}})
    sources={'data_manifest_sha256':sha256_file(manifest), 'model_manifest_sha256':'b'*64,
             'draft_manifest_sha256':'c'*64, 'policy_sha256':None, 'code':{'backend.py':'d'*64}}
    environment={'python':'3.13', 'os':'macOS-test', 'machine':'arm64', 'packages':{'mlx-vlm':'0.7.1'}}
    runtime={'module':{'path':'/historical/missing/upstream.py','sha256':'e'*64}}
    config={'stage':'collect','max_new_tokens':4,'warmup_tokens':4,'data_manifest':str(manifest),
            'model_path':str(tmp_path/'model'),'allow_battery':False,'min_battery':20}
    collect_rows=[row('confidence_collect',i,.8,2.5,True) for i in [10,11]]
    save_rows(collect/'samples.jsonl', collect_rows)
    collect_summary={'status':'completed', 'config':config, 'completed_measurements':2,
                     'methods':['confidence_collect'], 'source_row_indices':[10,11], 'sources':sources,
                     'samples_sha256':sha256_file(collect/'samples.jsonl'), 'environment':environment,
                     'method_metadata':{'confidence_collect':{'upstream_runtime_sources':runtime}}}
    write(collect/'summary.json',collect_summary)
    common={'selected_threshold':.5,'calibration_source_row_indices':[10,11],'expected_max_new_tokens':4,
            'source_provenance':sources,'source_summary_path':str(collect/'summary.json'),
            'source_summary_sha256':sha256_file(collect/'summary.json'),'source_samples_path':str(collect/'samples.jsonl'),
            'source_samples_sha256':sha256_file(collect/'samples.jsonl'),'selection':{'objective':'macro balanced accuracy'}}
    write(policy_dir/'calibration.json', {**common,'status':'completed'})
    write(policy_dir/'policy.json', {**common,'status':'frozen','calibration_report_sha256':sha256_file(policy_dir/'calibration.json')})
    rows=[row('ar_before'),row('stock_mtp',prefill=.6,end=2.3,speculative=True),
          row('confidence_collect',prefill=.8,end=2.5,speculative=True),row('confidence_gate',prefill=.7,end=3.4),
          row('ar_after',prefill=.9,end=3.1)]
    save_rows(evaluation/'samples.jsonl',rows)
    metadata={name:{'upstream_runtime_sources':deepcopy(runtime)} for name in ConfidenceReport.METHODS}
    metadata['confidence_gate'].update(confidence_policy='gate',confidence_threshold=.5)
    summary={'status':'completed', 'config':{**config,'stage':'evaluate'}, 'completed_measurements':5,
             'methods':list(ConfidenceReport.METHODS),'source_row_indices':[20], 'selected_threshold':.5,
             'sources':{**sources,'policy_sha256':sha256_file(policy_dir/'policy.json')},
             'samples_sha256':sha256_file(evaluation/'samples.jsonl'),'environment':environment,'method_metadata':metadata}
    write(evaluation/'summary.json',summary)
    class Tokenizer:
        def decode(self,ids,**kwargs):return ''.join(chr(65+t%26) for t in ids if t!=103)
    monkeypatch.setitem(sys.modules,'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    monkeypatch.setattr(BenchmarkReplayBuilder,'_verified_tokenizer_assets',staticmethod(lambda *a:None))
    return SimpleNamespace(root=tmp_path,run=evaluation,collect=collect,policy=policy_dir/'policy.json',rows=rows,
                           report=ConfidenceReport(evaluation,policy_dir/'policy.json'))


def update_summary(experiment, transform):
    path=experiment.run/'summary.json';summary=json.loads(path.read_text());transform(summary);write(path,summary)


def update_rows(experiment, transform):
    path=experiment.run/'samples.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()];transform(rows);save_rows(path,rows)
    update_summary(experiment,lambda summary:summary.update(samples_sha256=sha256_file(path)))


def refreeze(experiment, change_policy=lambda p:None, change_calibration=lambda c:None):
    policy=json.loads(experiment.policy.read_text())
    calibration_path=experiment.policy.with_name('calibration.json')
    calibration=json.loads(calibration_path.read_text())
    change_calibration(calibration);write(calibration_path,calibration)
    policy['calibration_report_sha256']=sha256_file(calibration_path)
    change_policy(policy);write(experiment.policy,policy)
    update_summary(experiment,lambda summary:summary['sources'].update(policy_sha256=sha256_file(experiment.policy)))


def test_build_uses_actual_acceptance_and_observed_eos_without_mutating_sources(experiment):
    files=[p for p in experiment.root.rglob('*') if p.is_file()]
    before={p:sha256_file(p) for p in files}
    result=experiment.report.build(experiment.root/'public')
    assert result['methods']['stock_mtp']['acceptance']['draft_acceptance_rate']==1.
    assert 'acceptance_rate' not in result['methods']['stock_mtp']['acceptance']
    assert result['methods']['stock_mtp']['first_eos_output_position_1based']==4
    assert result['methods']['stock_mtp']['first_eos_observed_seconds']==2.3
    assert result['heldout_shadow_classification']['confusion']['true_positive']==1
    trace=json.loads((experiment.root/'public/trace.json').read_text())
    assert '4 tokens' in trace['metadata']['label'] and '[10, 11]' in trace['metadata']['source']
    fallback=trace['lanes']['MTP + gate']['events'][1]
    assert fallback['round']==0 and fallback['confidence']['decision']=='fallback'
    assert fallback['confidence']['confidence_product']==.02
    assert before=={p:sha256_file(p) for p in files}


def test_all_five_use_same_window_and_preserve_pair_windows(experiment):
    stats=experiment.report.build(experiment.root/'public')['trajectory_statistics']
    assert stats['common_window']['start_seconds']==.9
    assert stats['common_window']['end_seconds']==2.3
    shared=stats['common_window_comparisons']
    assert set(shared)==set(ConfidenceReport.METHODS)
    assert {(s['window']['start_seconds'],s['window']['end_seconds']) for s in shared.values()}=={(.9,2.3)}
    assert stats['pair_window_comparisons']['stock_mtp']['window']['start_seconds']==.6
    # AR counts rise at 1 and 2; the MTP block arrives only at the excluded end.
    assert shared['stock_mtp']['time_weighted_lead_mean_tokens']==pytest.approx(-8/7)
    assert shared['stock_mtp']['time_weighted_lead_sd_tokens']==pytest.approx(math.sqrt(13)/7)
    assert shared['ar_before']['time_weighted_lead_sd_tokens']==0.


@pytest.mark.parametrize('mismatch',[False,True])
def test_pairs_are_accepted_by_renderer_without_rendering_gif(experiment,mismatch,monkeypatch):
    if mismatch:
        def change(rows):
            lane=rows[3]
            lane['generated_token_ids'][2]=777
            lane['generation_trace']['token_ids'][2]=777
            lane['generation_trace']['events'][2]['token_ids']=[777]
        update_rows(experiment,change)
    experiment.report.build(experiment.root/'public')
    from inference_lab.visualization.render import GenerationReplay, ReplayConfig
    # Avoid machine-specific font discovery; actual layout still executes.
    from PIL import ImageFont
    monkeypatch.setattr('inference_lab.visualization.render.Typography',lambda:lambda *a:ImageFont.load_default())
    for method in ('stock_mtp','confidence_gate'):
        pair=json.loads((experiment.root/f'public/{method}-replay.json').read_text())
        differs=mismatch and method=='confidence_gate'
        assert pair['parity']['equal'] is not differs
        assert pair['parity']['first_mismatch_index']==(2 if differs else None)
        assert GenerationReplay(pair,ReplayConfig(allow_mismatch=True)).equal is not differs
        if differs:
            with pytest.raises(ValueError,match='Output tokens differ'):GenerationReplay(pair,ReplayConfig())


@pytest.mark.parametrize('corruption',['sleep','assessment','power'])
def test_saved_valid_flag_cannot_hide_bad_host_evidence(experiment,corruption):
    def change(rows):
        obs=rows[0]['host_observation']
        if corruption=='sleep':obs.update(observation(sleep=5))
        elif corruption=='assessment':obs['sleep_assessment']['sleep_seconds']=9
        else:obs['power_after']['percent']=1
    update_rows(experiment,change)
    with pytest.raises(ValueError,match='sleep|power|Mach'):experiment.report.load()


@pytest.mark.parametrize('key',['data_manifest_sha256','model_manifest_sha256','draft_manifest_sha256','code'])
def test_rejects_cross_run_provenance_change(experiment,key):
    update_summary(experiment,lambda summary:summary['sources'].update({key:{'backend.py':'f'*64} if key=='code' else 'f'*64}))
    with pytest.raises(ValueError,match='provenance differs'):experiment.report.load()


def test_policy_threshold_must_match_frozen_calibration_report(experiment):
    refreeze(experiment,change_policy=lambda policy:policy.update(selected_threshold=.75))
    with pytest.raises(ValueError,match='differs from calibration report'):experiment.report.load()


def test_runtime_hashes_compared_without_accessing_old_machine_paths(experiment):
    experiment.report.load()  # Saved /historical/missing/... paths need not exist.
    update_summary(experiment,lambda summary:summary['method_metadata']['confidence_gate']['upstream_runtime_sources']['module'].update(sha256='f'*64))
    with pytest.raises(ValueError,match='inconsistent runtime'):experiment.report.load()


def test_actual_gate_must_use_frozen_threshold(experiment):
    update_summary(experiment,lambda summary:summary['method_metadata']['confidence_gate'].update(confidence_threshold=.1))
    with pytest.raises(ValueError,match='gate metadata'):experiment.report.load()


def test_hashed_calibration_raw_cannot_include_heldout_row(experiment):
    path=experiment.collect/'samples.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['source_row_index']=20;save_rows(path,rows)
    digest=sha256_file(path)
    summary_path=experiment.collect/'summary.json';summary=json.loads(summary_path.read_text())
    summary['samples_sha256']=digest;write(summary_path,summary)
    changes={'source_samples_sha256':digest,'source_summary_sha256':sha256_file(summary_path)}
    refreeze(experiment,lambda p:p.update(changes),lambda c:c.update(changes))
    with pytest.raises(ValueError,match='Calibration raw coverage'):experiment.report.load()


def test_complete_calibration_host_evidence_is_rechecked(experiment):
    path=experiment.collect/'samples.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['host_observation']['clock_after']['continuous_ticks']+=10
    save_rows(path,rows);summary_path=experiment.collect/'summary.json';summary=json.loads(summary_path.read_text())
    summary['samples_sha256']=sha256_file(path);write(summary_path,summary)
    changes={'source_samples_sha256':sha256_file(path),'source_summary_sha256':sha256_file(summary_path)}
    refreeze(experiment,lambda p:p.update(changes),lambda c:c.update(changes))
    with pytest.raises(ValueError,match='Mach'):experiment.report.load()


@pytest.mark.parametrize('corruption',['incomplete','count','prompt_hash','output_budget','software'])
def test_rejects_invalid_saved_evaluation(experiment,corruption):
    if corruption=='incomplete':update_summary(experiment,lambda s:s.update(status='running'))
    elif corruption=='count':update_summary(experiment,lambda s:s.update(completed_measurements=4))
    elif corruption=='prompt_hash':update_rows(experiment,lambda rows:rows[0].update(prompt_token_sha256='f'*64))
    elif corruption=='output_budget':update_rows(experiment,lambda rows:rows[0].update(decode_tokens=4))
    else:update_summary(experiment,lambda s:s['environment']['packages'].update({'mlx-vlm':'different'}))
    with pytest.raises(ValueError):experiment.report.load()


@pytest.mark.parametrize('corruption',[None,'hash','count','heldout','policy_metadata'])
def test_optional_prefreeze_round_dataset_is_hash_bound_and_disjoint(experiment,corruption):
    path=experiment.policy.with_name('rounds.jsonl')
    save_rows(path,[{'source_row_index':20 if corruption=='heldout' else 10,'round':1}])
    metadata={'schema_version':1,'filename':'rounds.jsonl','sha256':sha256_file(path),
              'count':2 if corruption=='count' else 1,'unit':'verified_full_two_proposal_round'}
    refreeze(experiment,lambda p:p.update(rounds_dataset=metadata),lambda c:c.update(rounds_dataset=metadata))
    if corruption=='hash':path.write_text(path.read_text()+'\n')
    if corruption=='policy_metadata':
        policy=json.loads(experiment.policy.read_text());policy['rounds_dataset']['count']=9
        write(experiment.policy,policy)
        update_summary(experiment,lambda s:s['sources'].update(policy_sha256=sha256_file(experiment.policy)))
    if corruption is None:experiment.report.load()
    else:
        with pytest.raises(ValueError,match='round dataset|hash mismatch'):experiment.report.load()
