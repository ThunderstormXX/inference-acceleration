"""Conservative surface-form audit of verified word-versus-digit rejections."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from inference_lab.visualization.replay_logs import BenchmarkReplayBuilder


class NumberFormAnalyzer:
    """Classify the first rejected token, never a discarded conditioned suffix."""
    WORDS = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
    WORDS.update(twenty=20, thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90,
                 hundred=100, thousand=1000, million=1000000, billion=1000000000)

    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    def text(self, ids):
        return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)

    @classmethod
    def form(cls, value):
        value=value.strip().casefold()
        if value in cls.WORDS:return ('word',cls.WORDS[value])
        if re.fullmatch(r'[0-9]+',value):return ('digits',int(value))
        return None

    @classmethod
    def prefix_form(cls, value):
        match=re.match(r'\s*([0-9]+|[A-Za-z]+)\b',value)
        return cls.form(match[1]) if match else None

    def analyze(self, records, generations):
        counts=Counter();examples=[];rejections=0;strict=0
        for record in records:
            accepted=record['accepted_count']
            if accepted==2:continue
            if accepted not in (0,1) or len(record['proposal_token_ids'])!=2:
                raise ValueError('Expected verified full two-proposal calibration pairs')
            index=record['source_row_index'];ids=generations[index]
            position=record['output_count_before']+accepted
            proposal_id=record['proposal_token_ids'][accepted]
            if position>=len(ids):raise ValueError('Missing true correction in generated output')
            correction_id=ids[position]
            if len(record['emitted_token_ids'])<=accepted or record['emitted_token_ids'][accepted]!=correction_id:
                raise ValueError('Correction differs from actual generation')
            if proposal_id==correction_id:raise ValueError('Rejected proposal equals correction')
            rejections+=1
            proposal=self.text([proposal_id]);correction=self.text([correction_id])
            left_token,right_token=self.form(proposal),self.form(correction)
            strict+=bool(left_token and right_token and left_token[0]!=right_token[0])
            proposal_prefix=self.text(record['proposal_token_ids'][accepted:])
            actual_continuation=self.text(ids[position:position+12])
            left,right=self.prefix_form(proposal_prefix),self.prefix_form(actual_continuation)
            if not left or not right or left[0]==right[0]:continue
            category=f'{left[0]}_to_{right[0]}'
            counts[category]+=1
            same=left[1]==right[1]
            counts['same_numeric_prefix_value']+=same
            examples.append({'source_row_index':index,'round':record['round'],'accepted_count':accepted,
                             'mismatch_output_index_0based':position,'category':category,
                             'proposal_token_id':proposal_id,'correction_token_id':correction_id,
                             'proposal_text':proposal,'correction_text':correction,
                             'same_numeric_prefix_value':same,
                             'proposal_prefix_text':proposal_prefix,'numeric_prefix_values':[left[1],right[1]],
                             'previous_context':self.text(ids[max(0,position-12):position]),
                             'actual_continuation':actual_continuation,
                             'p1':record['p1'],'p2':record['p2'],'confidence_product':record['confidence_product']})
        count=len(examples)
        return {'schema_version':1,'full_pairs':len(records),'rejected_pairs':rejections,
                'strict_single_token_mismatches':strict,'word_digit_surface_mismatches':count,'fraction_of_rejected_pairs':count/rejections if rejections else None,
                'counts':dict(counts),'examples':examples,
                'scope':'Calibration only; first mismatch of each verified full pair. Compare numeral prefixes of remaining proposals (up to2tokens) with actual committed continuation (up to12tokens), allowing leading whitespace. Strict one-token count is also reported.',
                'limits':'Surface heuristic, not semantic equivalence or complete error taxonomy. English number-word prefix or digit prefix only; compound numerals, punctuation and other languages may be missed or only partially read. Equal prefix values do not prove equal complete numbers; actual continuation is shown. Counts do not predict counterfactual speedup.'}


def build_report(policy_path, output):
    def verified(path,digest):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Input hash mismatch')
        return raw
    policy_path=Path(policy_path)
    policy=json.loads(policy_path.read_text())
    if policy['status']!='frozen':raise ValueError('Expected frozen calibration policy')
    flat=policy['rounds_dataset']
    records=[json.loads(line) for line in verified(policy_path.parent/flat['filename'],flat['sha256']).splitlines()]
    if len(records)!=flat['count']:raise ValueError('Round dataset count mismatch')
    rows=[json.loads(line) for line in verified(policy['source_samples_path'],policy['source_samples_sha256']).splitlines()]
    indices=policy['calibration_source_row_indices']
    if [r['source_row_index'] for r in rows]!=indices or any(r['source_row_index'] not in indices for r in records):
        raise ValueError('Calibration coverage mismatch or heldout leakage')
    summary=json.loads(verified(policy['source_summary_path'],policy['source_summary_sha256']))
    model=Path(summary['config']['model_path'])
    BenchmarkReplayBuilder._verified_tokenizer_assets(model,policy['source_provenance']['model_manifest_sha256'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    result=NumberFormAnalyzer(tokenizer).analyze(records,{r['source_row_index']:r['generated_token_ids'] for r in rows})
    result['provenance']={'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'rounds_sha256':flat['sha256'],
                          'samples_sha256':policy['source_samples_sha256'],'model_manifest_sha256':policy['source_provenance']['model_manifest_sha256']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);result=build_report(args.policy,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples','provenance')},ensure_ascii=False,indent=2))
    return 0
