"""Build an offline example atlas from hash-checked, fully verified MTP rounds."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import gzip
import json
import math
from pathlib import Path
import re

from inference_lab.core.config import ROOT
from inference_lab.core.io import sha256_file
from inference_lab.visualization.replay_logs import BenchmarkReplayBuilder
from inference_lab.visualization.trace import _enrich_events, display_tokens
from .report import ConfidenceReport


class ExampleCorpus:
    TAG_LABELS={
        'number':'Цифры в паре','latex':'LaTeX в ближайшем тексте',
        'math_symbol':'Математические символы','markup':'Markdown / маркеры',
        'newline':'Перенос строки','whitespace':'Разница в пробелах',
        'wording':'Разные буквенные фрагменты','short_fragment':'Короткий фрагмент слова',
        'high_confidence_failure':'Уверенный отказ (score ≥ 0,9)',
        'low_confidence_success':'Угадал при score ≤ 0,3',
    }

    def __init__(self,tokenizer,context_tokens=32,continuation_tokens=16):
        if context_tokens<1 or continuation_tokens<1:raise ValueError('Positive context lengths required')
        self.tokenizer=tokenizer;self.context_tokens=context_tokens;self.continuation_tokens=continuation_tokens

    def decode(self,ids):
        return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)

    @classmethod
    def tags(cls,item):
        tags=[]
        proposed=''.join(t['text'] for t in item['proposal'])
        actual=''.join(t['text'] for t in item['committed'])
        nearby=item['context'][-80:]+proposed+actual
        if re.search(r'\d',proposed+actual):tags.append('number')
        if re.search(r'\\[A-Za-z]+',nearby) or re.search(r'[{}]',proposed+actual):tags.append('latex')
        if re.search(r'[=+−^<>×÷]',proposed+actual):tags.append('math_symbol')
        if re.search(r'[*#`_]',proposed+actual):tags.append('markup')
        if '\n' in proposed+actual:tags.append('newline')
        if item['accepted']<2:
            left=item['proposal'][item['accepted']]['text']
            right=item['committed'][item['accepted']]['text']
            if left!=right and left.strip()==right.strip():tags.append('whitespace')
            if any(c.isalpha() for c in left) and any(c.isalpha() for c in right):tags.append('wording')
            if any(re.fullmatch(r'[A-Za-z]{1,3}',t) for t in (left,right)):tags.append('short_fragment')
            if item['score']>=.9:tags.append('high_confidence_failure')
        elif item['score']<=.3:tags.append('low_confidence_success')
        return tags

    def row_examples(self,row,split):
        if row.get('method')!='confidence_collect':raise ValueError('Use always-verify confidence observations only')
        lane=deepcopy(row['generation_trace']);ids=lane['token_ids']
        _enrich_events(lane,self.tokenizer)
        pieces=display_tokens(self.tokenizer,ids)['token_texts']
        drafts={e['round']:e for e in lane['events'] if e['type']=='draft'}
        commits={e['round']:e for e in lane['events'] if e['type']=='commit' and e.get('round',0)>0}
        result=[]
        for record in row['confidence_rounds']:
            if len(record['proposal_token_ids'])!=2:continue
            number=record['round'];draft=drafts[number];commit=commits[number]
            probabilities=[record.get(k) for k in ('p1','p2','confidence_product')]
            if any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probabilities) or not math.isclose(probabilities[0]*probabilities[1],probabilities[2],rel_tol=1e-10,abs_tol=1e-15):
                raise ValueError('Invalid pair probability evidence')
            accepted=record['accepted_count'];position=record['output_count_before']
            if (record['decision']!='verify' or type(accepted) is not int or not 0<=accepted<=2
                    or draft['token_ids']!=record['proposal_token_ids'] or commit['accepted_count']!=accepted
                    or record['emitted_token_ids']!=commit['token_ids'] or position!=draft['output_count']
                    or commit['token_ids']!=ids[position:position+len(commit['token_ids'])]):
                raise ValueError('Confidence round differs from actual traced generation')
            if accepted<2 and (len(commit['token_ids'])<=accepted or draft['token_ids'][accepted]==commit['token_ids'][accepted]):
                raise ValueError('Missing genuine target correction')
            start=max(0,position-self.context_tokens)
            # Stable display pieces reflect the complete true output, not decoded
            # verifier suffixes conditioned on a rejected draft token.
            context=''.join(pieces[start:position])
            item={
                'id':f"c{row['source_row_index']}-r{number}",'chain':row['source_row_index'],'split':split,
                'round':number,'position':position,'accepted':accepted,
                'p1':record['p1'],'p2':record['p2'],'score':record['confidence_product'],
                'context':context,'context_ids':ids[start:position],
                'proposal':[{'id':token,'text':text,'status':'accepted' if i<accepted else 'rejected' if i==accepted else 'discarded_suffix'}
                            for i,(token,text) in enumerate(zip(draft['token_ids'],draft['token_texts']))],
                'committed':[{'id':token,'text':text,'origin':'accepted_draft' if i<accepted else 'target'}
                             for i,(token,text) in enumerate(zip(commit['token_ids'],commit['token_texts']))],
                'actual_next':''.join(pieces[position:position+self.continuation_tokens]),
                'actual_next_ids':ids[position:position+self.continuation_tokens],
                'corrected_context_chars':max(draft['context_replaced_characters'],commit['context_replaced_characters']),
            }
            if item['corrected_context_chars']:
                item.update(proposal_context=draft['context_text'][-240:],committed_context=commit['context_text'][-240:])
            item['tags']=self.tags(item);result.append(item)
        return result

    @staticmethod
    def streaks(examples):
        result=[]
        by_chain=defaultdict(list)
        for item in examples:by_chain[item['chain']].append(item)
        for name,predicate in [('all_accepted',lambda r:r['accepted']==2),('none_accepted',lambda r:r['accepted']==0),('not_full_pair',lambda r:r['accepted']<2)]:
            for chain,rows in by_chain.items():
                current=[]
                for row in rows+[None]:
                    if row is not None and predicate(row) and (not current or row['round']==current[-1]['round']+1):
                        current.append(row);continue
                    if current:
                        result.append({'type':name,'chain':chain,'split':current[0]['split'],'length':len(current),
                                       'first_round':current[0]['round'],'last_round':current[-1]['round'],
                                       'ids':[r['id'] for r in current]})
                    current=[row] if row is not None and predicate(row) else []
        return sorted(result,key=lambda r:(-r['length'],r['chain'],r['first_round']))

    def build(self,rows,selection=None):
        examples=[];chains=[]
        for split,row in rows:
            examples.extend(self.row_examples(row,split))
            chains.append({'index':row['source_row_index'],'split':split,'problem':row['problem']})
        indexed={x['id']:x for x in examples}
        if len(indexed)!=len(examples):raise ValueError('Duplicate chain/round key')
        selected=[]
        if selection:
            for key,note in selection.get('notes',{}).items():
                if key not in indexed:raise ValueError(f'Unknown annotated example: {key}')
                indexed[key]['annotation']=note
            for section in selection['sections']:
                for key in section['ids']:
                    if key not in indexed:raise ValueError(f'Unknown selected example: {key}')
                    if key not in selected:selected.append(key)
        else:
            # Deterministic stratification by chain, outcome and score extremes.
            for chain in chains:
                for accepted in (2,1,0):
                    group=sorted((x for x in examples if x['chain']==chain['index'] and x['accepted']==accepted),key=lambda x:(x['score'],x['round']))
                    for row in group[:1]+group[-1:]:
                        if row['id'] not in selected:selected.append(row['id'])
        by_split={}
        for split in ('calibration','heldout'):
            counts=Counter(str(x['accepted']) for x in examples if x['split']==split)
            by_split[split]={'total':sum(counts.values()),'by_outcome':{str(i):counts[str(i)] for i in range(3)}}
        counts=Counter(str(x['accepted']) for x in examples)
        tag_stats=[]
        for tag,label in self.TAG_LABELS.items():
            group=[x for x in examples if tag in x['tags']]
            c=Counter(str(x['accepted']) for x in group)
            tag_stats.append({'tag':tag,'label':label,'count':len(group),'by_outcome':dict(c)})
        return {'schema_version':1,'summary':{'total':len(examples),'by_outcome':{str(i):counts[str(i)] for i in range(3)},'by_split':by_split},
                'chains':chains,'examples':examples,'tag_labels':self.TAG_LABELS,'tag_statistics':tag_stats,
                'curated_ids':selected,'selection':selection,'streaks':self.streaks(examples),
                'interpretation':'Accepted means matching the target greedy token IDs, not mathematical correctness. Tags are overlapping surface signals, not diagnosed causes. Heldout is descriptive only; no threshold retuning.',
                'display':{'preceding_tokens':self.context_tokens,'following_tokens':self.continuation_tokens,'gray_suffix':'Discarded after first mismatch; not independently labeled on the true continuation.'}}


class ExampleAtlas:
    def __init__(self,policy,heldout,selection=None):
        self.policy_path=Path(policy);self.heldout=Path(heldout);self.selection_path=Path(selection) if selection else None

    def build(self):
        summary,heldout,policy,_=ConfidenceReport(self.heldout,self.policy_path).load()
        calibration=[json.loads(line) for line in Path(policy['source_samples_path']).read_text().splitlines()]
        model=Path(summary['config']['model_path'])
        BenchmarkReplayBuilder._verified_tokenizer_assets(model,summary['sources']['model_manifest_sha256'])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        selection=json.loads(self.selection_path.read_text()) if self.selection_path else None
        corpus=ExampleCorpus(tokenizer).build([('calibration',r) for r in calibration]+[('heldout',r) for r in heldout if r['method']=='confidence_collect'],selection)
        corpus['provenance']={'policy_sha256':sha256_file(self.policy_path),'calibration_samples_sha256':policy['source_samples_sha256'],
                              'heldout_samples_sha256':summary['samples_sha256'],'model_manifest_sha256':summary['sources']['model_manifest_sha256'],
                              'selection_sha256':sha256_file(self.selection_path) if self.selection_path else None,
                              'builder_sha256':sha256_file(Path(__file__))}
        return corpus

    @staticmethod
    def code(text):
        if text and not text.strip():
            text=text.replace(' ','␠')
        text=text.replace('\n','↵').replace('\t','⇥')
        longest=max((len(m[0]) for m in re.finditer(r'`+',text)),default=0)
        fence='`'*(longest+1)
        return fence+' '+text+' '+fence

    @classmethod
    def markdown(cls,corpus):
        s=corpus['summary'];c=s['by_outcome'];lookup={x['id']:x for x in corpus['examples']}
        lines=['# Примеры MTP: где совпало, где разошлось','',
               f"В корпусе **{s['total']} проверенных пар** из 11 цепочек: **{c['2']}** приняты целиком, **{c['1']}** — только первый токен, **{c['0']}** — отказ уже на первом. Это два уже сохранённых набора: calibration100–109 и heldout110; повторного inference нет.",'',
               'В целиком пробельных токенах `␠` означает один пробел, `↵` — перенос, `⇥` — табуляцию. Успех означает совпадение с greedy target, а не правильность решения задачи. Серый суффикс после первого отказа не является независимо проверенной ошибкой. `p1 × p2` — score драфтера. Контекст и продолжение взяты из реально выданных токенов; предсказания verifier после ошибочного proposal не используются как продолжение.','',
               '[Интерактивный каталог всех примеров](mtp-examples.html). Откройте локальный HTML в браузере; GitHub показывает исходный файл. В каталоге есть поиск, фильтры, probabilities, token IDs и ссылки на отдельные раунды.','']
        sections=(corpus.get('selection') or {}).get('sections') or [{'title':'Подборка по цепочкам, результату и confidence','description':'Для каждой цепочки и каждого исхода взяты крайние значения score.','ids':corpus['curated_ids']}]
        for section in sections:
            lines+=['## '+section['title'],'',section.get('description',''),'']
            for key in section['ids']:
                x=lookup[key];marks={'accepted':'✅','rejected':'❌','discarded_suffix':'◻️'}
                proposed=' · '.join(marks[t['status']]+' '+cls.code(t['text']) for t in x['proposal'])
                actual=' · '.join(('✅' if t['origin']=='accepted_draft' else '🔵')+' '+cls.code(t['text']) for t in x['committed'])
                lines+=[f"### {key} · принято {x['accepted']}/2 · {x['split']}",'',
                        f"Контекст: {cls.code(x['context'])}",'',f'Драфт: {proposed}','',f'Реально выдано: {actual}','',
                        f"Продолжение: {cls.code(x['actual_next'])}",'',
                        f"p1={x['p1']:.4f}; p2={x['p2']:.4f}; score={x['score']:.4f}. Позиция: {x['position']} уже выданных токенов. [Открыть карточку](mtp-examples.html#{key}).",'']
                if x.get('annotation'):lines += [x['annotation'],'']
        lines+=['## Как интерпретировать подборку','',
                'Тематические группы подобраны для изучения, не являются случайной выборкой и не оценивают частоту причин. Автоматические tags — пересекающиеся признаки текста. Разница в формулировке, пунктуации или LaTeX может сохранять смысл, но это не доказано сравнением одного токена. Для дообучения нужны отдельные train/validation/test по целым задачам; уже просмотренный heldout нельзя снова назвать слепым тестом.','',
                'Скрипт: `bash scripts/report/mtp_examples.sh`. JSON.gz и его SHA-256 записываются рядом с HTML. Параметры и локальный запуск описаны в [руководстве](../mtp-examples-guide.md).','']
        return '\n'.join(lines)

    def write(self,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        corpus=self.build();raw=json.dumps(corpus,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        template=(ROOT/'src/inference_lab/visualization/templates/mtp_examples.html').read_text()
        if template.count('__MTP_EXAMPLE_DATA__')!=1:raise ValueError('Template requires exactly one data placeholder')
        safe=raw.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
        (output/'mtp-examples.json.gz').write_bytes(gzip.compress((raw+'\n').encode(),compresslevel=9,mtime=0))
        (output/'mtp-examples.html').write_text(template.replace('__MTP_EXAMPLE_DATA__',safe))
        (output/'mtp-examples.md').write_text(self.markdown(corpus))
        manifest={'schema_version':1,'summary':corpus['summary'],'curated_count':len(corpus['curated_ids']),
                  'provenance':corpus['provenance'],'template_sha256':sha256_file(ROOT/'src/inference_lab/visualization/templates/mtp_examples.html'),
                  'files':{name:sha256_file(output/name) for name in ('mtp-examples.json.gz','mtp-examples.html','mtp-examples.md')}}
        (output/'mtp-examples.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,default=ROOT/'artifacts/experiments/mtp-confidence/policy2048/policy.json')
    p.add_argument('--heldout',type=Path,default=ROOT/'artifacts/experiments/mtp-confidence/heldout2048')
    p.add_argument('--selection',type=Path,default=ROOT/'configs/reports/mtp-examples.json')
    p.add_argument('--output',type=Path,default=ROOT/'docs/examples')
    args=p.parse_args(argv)
    print(json.dumps(ExampleAtlas(args.policy,args.heldout,args.selection).write(args.output),indent=2))
    return 0
