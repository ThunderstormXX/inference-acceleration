from copy import deepcopy
import math
import pytest
from inference_lab.experiments.confidence.examples import ExampleCorpus,ExampleAtlas


class Tokenizer:
    tokens={1:'Start ',2:'wrong',3:' suffix',4:'real',5:' COUNTERFACTUAL',6:' NEVER',7:' next',8:' words',9:'.'}
    def decode(self,ids,**kwargs):return ''.join(self.tokens[t] for t in ids)


def row():
    records=[{'round':1,'output_count_before':1,'proposal_token_ids':[2,3],'decision':'verify','accepted_count':0,'emitted_token_ids':[4],'target_token_ids':[4,5,6],'p1':.99,'p2':.95,'confidence_product':.9405},
             {'round':2,'output_count_before':2,'proposal_token_ids':[7,8],'decision':'verify','accepted_count':2,'emitted_token_ids':[7,8,9],'target_token_ids':[7,8,9],'p1':.1,'p2':.2,'confidence_product':.02}]
    events=[{'type':'commit','round':0,'token_ids':[1],'output_count':1,'t':.1}]
    for r in records:
        events += [{'type':'draft','round':r['round'],'token_ids':r['proposal_token_ids'],'output_count':r['output_count_before'],'t':r['round']},
                   {'type':'commit','round':r['round'],'token_ids':r['emitted_token_ids'],'output_count':r['output_count_before']+len(r['emitted_token_ids']),'accepted_count':r['accepted_count'],'t':r['round']+.5}]
    return {'method':'confidence_collect','source_row_index':100,'problem':'Test problem','confidence_rounds':records,
            'generation_trace':{'token_ids':[1,4,7,8,9],'events':events}}


def test_actual_continuation_is_not_counterfactual_verifier_suffix():
    examples=ExampleCorpus(Tokenizer()).row_examples(row(),'calibration')
    first=examples[0]
    assert first['actual_next']=='real next words.'
    assert [t['status'] for t in first['proposal']]==['rejected','discarded_suffix']
    assert first['committed']==[{'id':4,'text':'real','origin':'target'}]
    assert 'COUNTERFACTUAL' not in str(first)
    assert 'high_confidence_failure' in first['tags']
    second=examples[1]
    assert all(t['status']=='accepted' for t in second['proposal'])
    assert [t['origin'] for t in second['committed']]==['accepted_draft','accepted_draft','target']
    assert 'low_confidence_success' in second['tags']


def test_partial_acceptance_and_true_context():
    r=row();r['generation_trace']['token_ids']=[1,2,4,7,8,9]
    first,second=r['confidence_rounds'];first.update(accepted_count=1,emitted_token_ids=[2,4]);second['output_count_before']=3
    ev=r['generation_trace']['events'];ev[2].update(token_ids=[2,4],output_count=3,accepted_count=1);ev[3]['output_count']=3;ev[4]['output_count']=6
    examples=ExampleCorpus(Tokenizer(),context_tokens=2,continuation_tokens=2).row_examples(r,'calibration')
    assert [t['status'] for t in examples[0]['proposal']]==['accepted','rejected']
    assert examples[1]['context_ids']==[2,4]
    assert examples[1]['context']=='wrongreal'
    assert examples[0]['actual_next_ids']==[2,4]


@pytest.mark.parametrize('edit',[lambda r:r.update(decision='fallback'),lambda r:r.update(accepted_count=2),lambda r:r.update(confidence_product=.1),lambda r:r.update(p1=math.nan)])
def test_invalid_rounds_fail(edit):
    r=row();edit(r['confidence_rounds'][0])
    with pytest.raises(ValueError):ExampleCorpus(Tokenizer()).row_examples(r,'calibration')


def test_selection_and_counts_preserve_split():
    selection={'sections':[{'title':'Chosen','ids':['c100-r1','c100-r2']}],'notes':{'c100-r1':'A note'}}
    corpus=ExampleCorpus(Tokenizer()).build([('calibration',row())],selection)
    assert corpus['curated_ids']==['c100-r1','c100-r2']
    assert corpus['summary']['by_outcome']=={'0':1,'1':0,'2':1}
    assert corpus['summary']['by_split']['heldout']['total']==0
    assert corpus['examples'][0]['annotation']=='A note'
    assert 'A note' in ExampleAtlas.markdown(corpus)


def test_unknown_selection_fails():
    with pytest.raises(ValueError):ExampleCorpus(Tokenizer()).build([('calibration',row())],{'sections':[{'ids':['c110-r1']}]})


def test_streaks_break_on_gaps_and_chains():
    rows=[{'id':f'c{chain}-r{rnd}','chain':chain,'split':'calibration','round':rnd,'accepted':accepted} for chain,rnd,accepted in [(100,1,0),(100,2,0),(100,4,0),(101,1,0),(101,2,2),(101,3,2)]]
    zero=[s for s in ExampleCorpus.streaks(rows) if s['type']=='none_accepted']
    assert max(s['length'] for s in zero)==2
    assert zero[0]['ids']==['c100-r1','c100-r2']


def test_markdown_code_handles_backticks_and_linebreak():
    assert ExampleAtlas.code('a`b\nc')=='`` a`b↵c ``'
    assert ExampleAtlas.code('   ')=='` ␠␠␠ `'
    assert ExampleAtlas.code('       ')=='` ␠␠␠␠␠␠␠ `'
    assert ExampleAtlas.code('\n  ')=='` ↵␠␠ `'


def test_html_embedded_data_cannot_close_script(monkeypatch,tmp_path):
    import gzip,json,re
    corpus=ExampleCorpus(Tokenizer()).build([('calibration',row())])
    corpus['chains'][0]['problem']='</script><script>alert("x")</script>'
    corpus['provenance']={}
    atlas=ExampleAtlas('unused-policy','unused-heldout')
    monkeypatch.setattr(atlas,'build',lambda:corpus)
    manifest=atlas.write(tmp_path)
    html=(tmp_path/'mtp-examples.html').read_text()
    embedded=re.search(r'<script type="application/json" id="example-data">(.*?)</script>',html,re.S)[1]
    assert '<script>' not in embedded
    assert json.loads(embedded)==corpus
    assert json.loads(gzip.decompress((tmp_path/'mtp-examples.json.gz').read_bytes()))==corpus
    assert manifest['curated_count']==2
