"""CPU regression checks for durable, provenance-checked confidence experiments."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import inference_lab.experiments.confidence.runner as runner_module
from inference_lab.experiments.confidence.runner import (
    ConfidenceExperimentConfig, ConfidenceExperimentRunner, RequestGuard,
)
from inference_lab.core.host_clock import MacSleepClock
from inference_lab.core.io import sha256_file


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def observation(sleep=0, battery=False, percent=90, allow_battery=False):
    before = dict(schema_version=1, absolute_ticks=100, continuous_ticks=200,
                  timebase_numer=1_000_000_000, timebase_denom=1, sampling_span_ticks=0)
    after = {**before, 'absolute_ticks':102, 'continuous_ticks':202 + sleep}
    raw = f"Now drawing from '{'Battery Power' if battery else 'AC Power'}'\n -InternalBattery-0 {percent}%; {'discharging' if battery else 'charging'};"
    power = RequestGuard.parse_power(raw, allow_battery, 20)
    assessment = MacSleepClock.assess(before, after)
    return {'clock_before':before, 'clock_after':after, 'sleep_assessment':assessment,
            'power_before':deepcopy(power), 'power_after':deepcopy(power),
            'valid':not assessment['sleep_detected'] and power['allowed']}


def measurement(n, prompt_count=3):
    ids = list(range(1000, 1000+n))
    return {'prompt_tokens':prompt_count, 'generated_tokens':n, 'decode_tokens':n-1,
            'prefill_seconds':.5, 'decode_seconds':1.5, 'generated_token_ids':ids,
            'generation_trace':{'schema_version':1, 'token_ids':ids, 'eos_token_ids':[],
                'prefill_seconds':.5, 'decode_seconds':1.5, 'total_seconds':2.,
                'instrumentation':{'enabled':True},
                'events':[{'type':'commit', 't':.5+i/n, 'token_ids':[token], 'output_count':i+1}
                          for i,token in enumerate(ids)]}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path/'artifacts').mkdir()
    model, draft = tmp_path/'model', tmp_path/'draft'
    model.mkdir(); draft.mkdir()
    write(model/'download-manifest.json', {'revision':'target'})
    write(draft/'download-manifest.json', {'revision':'draft'})
    calibration, heldout = tmp_path/'calibration.jsonl', tmp_path/'heldout.jsonl'
    calibration.write_text(''.join(json.dumps({'source_row_index':i,'problem':'p'})+'\n' for i in [100,101]))
    heldout.write_text(json.dumps({'source_row_index':110,'problem':'heldout'})+'\n')
    manifest=tmp_path/'manifest.json'
    write(manifest, {'status':'completed','source_revision':'revision','splits':{
        key:{'path':str(path),'sha256':sha256_file(path),'source_row_indices':indices}
        for key,path,indices in [('calibration',calibration,[100,101]),('heldout',heldout,[110])]}})
    state={'calls':0, 'finishes':0, 'short':{}, 'fail_on':None, 'sleep_on':None,
           'metadata':{'framework':'cpu-fixture'},
           'environment':{'python':'3.13','os':'macOS-test','machine':'arm64','packages':{'mlx':'test'},'pid':1}}
    class Guard(RequestGuard):
        def __init__(self,*args):pass
        def start(self):return {}
        def finish(self, value):
            state['finishes'] += 1
            value.update(observation(sleep=5 if state['finishes']==state['sleep_on'] else 0))
            return value
    class Backend:
        tokenizer=object()
        _mx=SimpleNamespace(synchronize=lambda:None, clear_cache=lambda:None)
        def load(self):pass
        def metadata(self):return deepcopy(state['metadata'])
        def measure(self,prompt,n):
            state['calls'] += 1
            if state['calls']==state['fail_on']:raise RuntimeError('fixture interruption')
            return measurement(state['short'].get(state['calls'], n),len(prompt))
    monkeypatch.setattr(runner_module,'ROOT',tmp_path)
    monkeypatch.setattr(runner_module,'RequestGuard',Guard)
    monkeypatch.setattr(runner_module,'environment',lambda:deepcopy(state['environment']))
    monkeypatch.setattr(runner_module,'resource_snapshot',lambda:{})
    monkeypatch.setattr(runner_module,'build_prompt',lambda *a,**kw:[1,2,3])
    monkeypatch.setattr(ConfidenceExperimentRunner,'sources',lambda self:{'runner.py':'a'*64})
    monkeypatch.setattr(ConfidenceExperimentRunner,'backend',lambda self,*a:Backend())
    config=ConfidenceExperimentConfig(stage='collect',output=str(tmp_path/'out'),data_manifest=str(manifest),
        model_path=str(model),draft_path=str(draft),max_new_tokens=8,warmup_tokens=4)
    return SimpleNamespace(root=tmp_path,state=state,config=config)


def run(setup, resume=False):
    return ConfidenceExperimentRunner(setup.config,resume=resume).run()


def load_summary(setup):return json.loads((Path(setup.config.output)/'summary.json').read_text())


def rewrite_samples(setup, transform):
    directory=Path(setup.config.output)
    rows=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
    transform(rows)
    (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary=load_summary(setup)
    summary['samples_sha256']=sha256_file(directory/'samples.jsonl')
    write(directory/'summary.json',summary)


def policy_fixture(setup):
    run(setup)
    summary=load_summary(setup)
    directory=setup.root/'policy';directory.mkdir()
    source=Path(setup.config.output)
    common={'schema_version':1,'selected_threshold':.5,'calibration_source_row_indices':[100,101],
        'expected_max_new_tokens':8,'source_provenance':summary['sources'],
        'source_samples_path':str(source/'samples.jsonl'),'source_samples_sha256':sha256_file(source/'samples.jsonl'),
        'source_summary_path':str(source/'summary.json'),'source_summary_sha256':sha256_file(source/'summary.json'),
        'selection':{'objective':'fixture'}}
    write(directory/'calibration.json',{**common,'status':'completed'})
    policy={**common,'status':'frozen','calibration_report_sha256':sha256_file(directory/'calibration.json')}
    path=directory/'policy.json';write(path,policy)
    config=replace(setup.config,stage='evaluate',output=str(setup.root/'evaluation'),policy_path=str(path))
    return config,path,policy


def test_valid_collect_resume_and_mutable_host_fields(setup):
    run(setup)
    assert load_summary(setup)['completed_measurements']==2
    setup.state['environment'].update(pid=999,power_source={'raw':'changed'},thermal_state='different')
    calls=setup.state['calls']
    run(setup,resume=True)
    assert setup.state['calls']==calls
    assert load_summary(setup)['status']=='completed'


@pytest.mark.parametrize('call', [1,2])
def test_short_warmup_or_fresh_row_cannot_complete(setup,call):
    setup.state['short'][call]=2
    with pytest.raises(ValueError,match='exact output budget'):run(setup)
    assert load_summary(setup)['status']=='failed'
    assert not (Path(setup.config.output)/'samples.jsonl').exists()


def test_short_saved_row_rejected_even_with_valid_hash(setup):
    run(setup)
    rewrite_samples(setup,lambda rows:rows[0].update(measurement(7)))
    with pytest.raises(ValueError,match='exact output budget'):run(setup,resume=True)


def test_invalid_sleep_request_retained_outside_samples(setup):
    setup.state['sleep_on']=2
    with pytest.raises(RuntimeError,match='Request crossed sleep'):run(setup)
    output=Path(setup.config.output)
    invalid=json.loads((output/'invalid-samples.jsonl').read_text())
    assert invalid['host_observation']['sleep_assessment']['sleep_detected']
    assert not (output/'samples.jsonl').exists()


@pytest.mark.parametrize('corruption', ['sleep','assessment','power'])
def test_saved_host_evidence_recomputed_not_trusted(setup,corruption):
    run(setup)
    def change(rows):
        obs=rows[0]['host_observation']
        if corruption=='sleep':obs.update(observation(sleep=5));obs['valid']=True
        elif corruption=='assessment':obs['sleep_assessment']['sleep_seconds']=9
        else:obs['power_after']['percent']=1
    rewrite_samples(setup,change)
    with pytest.raises(ValueError,match='sleep|power|Mach'):run(setup,resume=True)


def test_power_policy_applies_to_raw_both_endpoints():
    battery=observation(battery=True,percent=70,allow_battery=True)
    RequestGuard.validate(battery,True,20)
    with pytest.raises(ValueError,match='power'):RequestGuard.validate(battery,False,20)
    low=observation(battery=True,percent=10,allow_battery=True)
    low['valid']=True
    with pytest.raises(ValueError,match='power'):RequestGuard.validate(low,True,20)


def test_resume_rejects_package_version_change(setup):
    run(setup)
    setup.state['environment']['packages']['mlx']='different'
    with pytest.raises(ValueError,match='software environment changed'):run(setup,resume=True)


def test_resume_rejects_changed_installed_source_hash(setup):
    source=setup.root/'upstream.py';source.write_text('first')
    setup.state['metadata']['upstream_runtime_sources']={'module':{'path':str(source),'sha256':sha256_file(source)}}
    run(setup)
    source.write_text('different')
    with pytest.raises(ValueError,match='runtime source hash changed'):run(setup,resume=True)


def test_resume_preserves_differing_method_metadata(setup):
    setup.state['fail_on']=3
    with pytest.raises(RuntimeError,match='fixture interruption'):run(setup)
    old=deepcopy(load_summary(setup)['method_metadata'])
    setup.state['fail_on']=None
    setup.state['metadata']['different_runtime']='new'
    with pytest.raises(ValueError,match='refusing to overwrite'):run(setup,resume=True)
    assert load_summary(setup)['method_metadata']==old


def test_saved_warmup_budget_checked(setup):
    run(setup)
    summary=load_summary(setup);summary['warmups'][0]['generated_tokens']=2
    write(Path(setup.config.output)/'summary.json',summary)
    with pytest.raises(ValueError,match='Saved warmup'):run(setup,resume=True)


def test_valid_frozen_policy_and_evaluation(setup):
    config,_,_=policy_fixture(setup)
    rows,_,policy=ConfidenceExperimentRunner(config).inputs()
    assert [r['source_row_index'] for r in rows]==[110]
    assert policy['selected_threshold']==.5
    ConfidenceExperimentRunner(config).run()
    summary=json.loads((Path(config.output)/'summary.json').read_text())
    assert summary['completed_measurements']==5
    assert summary['parity']['110']['equal'] is True


@pytest.mark.parametrize('key',['data_manifest_sha256','model_manifest_sha256','draft_manifest_sha256','code'])
def test_policy_rejects_other_data_weights_or_code(setup,key):
    config,path,policy=policy_fixture(setup)
    policy['source_provenance'][key]='b'*64
    write(path,policy)
    with pytest.raises(ValueError,match='provenance|measurement code'):ConfidenceExperimentRunner(config).inputs()


@pytest.mark.parametrize('kind',['samples','summary','report'])
def test_policy_rejects_modified_calibration_file(setup,kind):
    config,path,policy=policy_fixture(setup)
    target=path.parent/'calibration.json' if kind=='report' else Path(policy[f'source_{kind}_path'])
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash mismatch'):ConfidenceExperimentRunner(config).inputs()


def test_policy_rejects_threshold_changed_after_freeze(setup):
    config,path,policy=policy_fixture(setup)
    policy['selected_threshold']=.9;write(path,policy)
    with pytest.raises(ValueError,match='selected_threshold'):ConfidenceExperimentRunner(config).inputs()


def test_heldout_rejects_calibration_software_change(setup):
    config,_,_=policy_fixture(setup)
    setup.state['environment']['packages']['mlx']='other'
    with pytest.raises(ValueError,match='held-out software'):ConfidenceExperimentRunner(config).run()


@pytest.mark.parametrize('kind', ['sleep', 'short'])
def test_invalid_warmup_preserved_separately_and_resume_rewarms(setup, kind):
    if kind == 'sleep': setup.state['sleep_on'] = 1
    else: setup.state['short'][1] = 2
    with pytest.raises((ValueError, RuntimeError)):
        run(setup)
    summary = load_summary(setup)
    assert len(summary['invalid_warmups']) == 1
    assert summary['warmups'] == []
    run(setup, resume=True)
    summary = load_summary(setup)
    assert summary['status'] == 'completed'
    assert len(summary['invalid_warmups']) == 1
    assert len(summary['warmups']) == 1
