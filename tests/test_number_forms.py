import pytest
from inference_lab.experiments.confidence.number_forms import NumberFormAnalyzer


class Tokenizer:
    pieces={1:'ctx',2:'four',3:'4',4:' ',5:'five',6:'5',7:' next',8:'fort',9:'40',10:'two',11:'2',12:'0'}
    def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)


def record(accepted=0,proposal=(2,7),emitted=(3,),position=1):
    return {'source_row_index':100,'round':1,'output_count_before':position,'proposal_token_ids':list(proposal),
            'emitted_token_ids':list(emitted),'accepted_count':accepted,'p1':.5,'p2':.8,'confidence_product':.4}


def test_word_to_digit_first_mismatch_only():
    result=NumberFormAnalyzer(Tokenizer()).analyze([record()],{100:[1,3,7]})
    assert result['word_digit_surface_mismatches']==1
    assert result['counts']=={'word_to_digits':1,'same_numeric_prefix_value':1}
    assert result['examples'][0]['actual_continuation']=='4 next'
    assert result['examples'][0]['previous_context']=='ctx'


def test_second_proposal_and_reverse_direction():
    row=record(accepted=1,proposal=(3,6),emitted=(3,5))
    result=NumberFormAnalyzer(Tokenizer()).analyze([row],{100:[1,3,5,7]})
    example=result['examples'][0]
    assert example['category']=='digits_to_word'
    assert example['mismatch_output_index_0based']==2
    assert example['previous_context']=='ctx4'
    assert example['correction_text']=='five'


def test_non_numeral_fragment_not_counted_and_success_excluded():
    rows=[record(proposal=(8,7)),record(accepted=2)]
    result=NumberFormAnalyzer(Tokenizer()).analyze(rows,{100:[1,3,7]})
    assert result['rejected_pairs']==1
    assert result['word_digit_surface_mismatches']==0


def test_digit_can_be_prefix_not_claimed_semantic_equivalence():
    result=NumberFormAnalyzer(Tokenizer()).analyze([record(proposal=(10,7),emitted=(11,))],{100:[1,11,12,7]})
    assert result['examples'][0]['same_numeric_prefix_value'] is False
    assert result['examples'][0]['actual_continuation']=='20 next'
    assert 'not semantic equivalence' in result['limits']


@pytest.mark.parametrize('ids,emitted', [([1,3],(5,)),([1,2],(2,))])
def test_wrong_correction_or_equal_rejection_fails(ids,emitted):
    with pytest.raises(ValueError):NumberFormAnalyzer(Tokenizer()).analyze([record(emitted=emitted)],{100:ids})


@pytest.mark.parametrize('text,result',[(' FOUR ',('word',4)),('004',('digits',4)),('four,',None),('два',None),('twenty one',None),('-4',None)])
def test_conservative_forms(text,result):
    assert NumberFormAnalyzer.form(text)==result


def test_whitespace_token_before_digit_is_detected():
    row=record(emitted=(4,))
    result=NumberFormAnalyzer(Tokenizer()).analyze([row],{100:[1,4,3,7]})
    assert result['strict_single_token_mismatches']==0
    assert result['word_digit_surface_mismatches']==1
    assert result['examples'][0]['same_numeric_prefix_value'] is True
