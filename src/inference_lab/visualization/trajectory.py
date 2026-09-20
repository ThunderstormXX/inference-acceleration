"""Plot observed token-count trajectories, signed lead and real rejection examples."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import pstdev


class TokenTrajectory:
    """Right-continuous counts from validated observed commits, never interpolated."""
    def __init__(self, lane):
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        def seconds(value, name):
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    f'Invalid {name} clock')
            return value
        def ids(value, name):
            require(isinstance(value, list) and bool(value)
                    and all(type(token) is int and token >= 0 for token in value),
                    f'{name} requires nonempty nonnegative integer token IDs')
            return value
        require(isinstance(lane, dict), 'Trajectory lane must be an object')
        self.lane = lane
        self.ids = ids(lane.get('token_ids'), 'Lane')
        self.prefill = seconds(lane.get('prefill_seconds'), 'prefill')
        decode = seconds(lane.get('decode_seconds'), 'decode')
        self.total = seconds(lane.get('total_seconds'), 'total')
        require(math.isclose(self.prefill + decode, self.total, rel_tol=1e-9, abs_tol=1e-9),
                'Total duration differs from prefill plus decode')
        events = lane.get('events')
        require(isinstance(events, list) and bool(events), 'Trajectory requires events')
        self.times, self.counts = [0.], [0]
        committed, previous, pending = [], 0., None
        for event in events:
            require(isinstance(event, dict) and event.get('type') in ('draft', 'commit'),
                    'Unknown trajectory event type')
            current = seconds(event.get('t'), 'event')
            require(previous <= current <= self.total + 1e-9, 'Events exceed duration or are not chronological')
            previous = current
            tokens = ids(event.get('token_ids'), 'Event')
            count = event.get('output_count')
            require(type(count) is int, 'Event output_count must be an integer')
            if event['type'] == 'draft':
                require(bool(committed) and pending is None and count == len(committed)
                        and current >= self.prefill, 'Draft has invalid count, phase, or ordering')
                pending = event
                continue
            if not committed:
                require(len(tokens) == 1 and current <= self.prefill + 1e-9,
                        'Prefill must commit exactly the first token before its phase ends')
            else:
                require(current >= self.prefill, 'Decode commit precedes prefill boundary')
            if pending is not None:
                proposed, accepted = pending['token_ids'], event.get('accepted_count')
                require(type(accepted) is int and 0 <= accepted <= len(proposed)
                        and event.get('draft_count') == len(proposed)
                        and event.get('round') == pending.get('round')
                        and event.get('proposed_token_ids') == proposed
                        and event.get('rejected_token_ids') == proposed[accepted:],
                        'Draft and verification counters disagree')
                require(tokens[:min(accepted, len(tokens))] == proposed[:min(accepted, len(tokens))],
                        'Accepted proposal prefix differs from committed tokens')
                pending = None
            else:
                require(not event.get('proposed_token_ids') and not event.get('accepted_count', 0),
                        'Speculative commit is missing its draft')
            committed.extend(tokens)
            require(count == len(committed), 'Commit output_count differs from cumulative token IDs')
            self.times.append(current)
            self.counts.append(count)
        require(pending is None, 'Trajectory ends with unverified draft proposals')
        require(committed == self.ids, 'Events disagree with generated IDs')
        self.end = self.times[-1]

    def step_points(self, end=None):
        """Extend the completed lane horizontally to a common plot endpoint."""
        end = self.end if end is None else end
        if type(end) not in (int, float) or not math.isfinite(end) or end < self.end:
            raise ValueError('Plot endpoint must be finite and not precede completion')
        return (self.times + [end], self.counts + [len(self.ids)]) if end > self.end else (list(self.times), list(self.counts))
    def count(self,t):
        return self.counts[max(0,bisect_right(self.times,t)-1)]
    def arrival_times(self):
        result=[]; previous=0
        for t,count in zip(self.times[1:],self.counts[1:]):
            result.extend([t]*(count-previous));previous=count
        return result
    def compare(self,other,*,window=None):
        start=max(self.prefill,other.prefill);end=min(self.end,other.end)
        definition='After both prefills, until first lane finishes; excludes final catch-up tail'
        if window is not None:
            if (not isinstance(window,(tuple,list)) or len(window)!=2
                    or any(type(value) not in (int,float) or not math.isfinite(value) for value in window)
                    or not start<=window[0]<window[1]<=end):
                raise ValueError('Explicit comparison window must be inside the common decode interval')
            start,end=window
            definition='Explicit shared decode window inside both lanes; excludes prefill and final catch-up tail'
        if end<=start:raise ValueError('No common decode interval')
        boundaries=sorted({start,end,*[t for t in self.times+other.times if start<t<end]})
        intervals=[(a,b,self.count(a)-other.count(a)) for a,b in zip(boundaries,boundaries[1:])]
        duration=end-start
        avg=sum((b-a)*v for a,b,v in intervals)/duration
        sd=math.sqrt(sum((b-a)*(v-avg)**2 for a,b,v in intervals)/duration)
        grid=[start+i*.1 for i in range(int(duration/.1)+1)]
        leads=[self.count(t)-other.count(t) for t in grid]
        increments=[b-a for a,b in zip(leads,leads[1:])]
        arrivals,reference=self.arrival_times(),other.arrival_times()
        equal=self.ids==other.ids
        return {'exact_token_parity':equal,'visible_completion_seconds':self.end,'baseline_completion_seconds':other.end,
                'same_output_completion_ratio':other.end/self.end if equal else None,
                'window':{'start_seconds':start,'end_seconds':end,'definition':definition},
                'lead_definition':'method committed tokens minus baseline committed tokens at same request-relative time',
                'time_weighted_lead_mean_tokens':avg,'time_weighted_lead_sd_tokens':sd,
                'lead_min_tokens':min(v for _,_,v in intervals),'lead_max_tokens':max(v for _,_,v in intervals),
                'lead_increment_sd_per_100ms_tokens':pstdev(increments) if increments else None,
                'jitter_caveat':'SD of signed-lead increments on a fixed100ms grid; descriptive pacing variability, not repeated-run uncertainty.',
                'time_saved_to_each_token_seconds':[a-b for a,b in zip(reference,arrivals)] if equal else None}


class TrajectoryReport:
    COLORS=['#26a69a','#e9b44c','#a68af9','#e67e8a','#7ea5e8']
    def __init__(self,trace):
        self.trace=trace
        self.lanes=trace.get('lanes') or {k:trace[k] for k in ('baseline','mtp')}
        if 'baseline' not in self.lanes:raise ValueError('Need a baseline trajectory')
        self.paths={k:TokenTrajectory(v) for k,v in self.lanes.items()}
    def write(self,output):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                             'axes.titleweight':'bold','figure.facecolor':'#fbfcfe','axes.facecolor':'#fbfcfe','savefig.facecolor':'#fbfcfe'})
        fig,axes=plt.subplots(3,1,figsize=(13,11),gridspec_kw={'height_ratios':[1.6,1,1]},layout='constrained')
        base=self.paths['baseline'];names=[k for k in self.paths if k!='baseline']
        plot_end=max(lane.end for lane in self.paths.values())
        axes[0].step(*base.step_points(plot_end),where='post',label='Baseline AR',color='#4263a6',lw=2)
        stats={}
        for i,name in enumerate(names):
            lane=self.paths[name];color=self.COLORS[i%len(self.COLORS)]
            axes[0].step(*lane.step_points(plot_end),where='post',label=name,color=color,lw=1.6)
            merged=sorted(set(base.times+lane.times+[plot_end]))
            axes[1].step(merged,[lane.count(t)-base.count(t) for t in merged],where='post',color=color,label=name,lw=1.2)
            stats[name]=lane.compare(base)
            saved=stats[name]['time_saved_to_each_token_seconds']
            if saved:
                axes[2].plot(range(1,len(saved)+1),saved,color=color,label=name,lw=1.2)
            else:
                axes[2].text(.03,.85-i*.1,f'{name}: output IDs differ; same-token latency curve omitted',transform=axes[2].transAxes,color=color)
        axes[0].set(title='Observed generation trajectory',ylabel='Committed output tokens',xlabel='Seconds from request start (prefill included)')
        axes[0].legend(loc='upper left');axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].axhline(0,color='#687785',lw=.8);axes[1].set(title='Signed token lead — positive means ahead of baseline',ylabel='Tokens ahead / behind',xlabel='Seconds from request start')
        axes[2].axhline(0,color='#687785',lw=.8);axes[2].set(title='Latency saved to the same output token — positive means earlier',ylabel='Baseline time − method time, s',xlabel='Output token index (1-based)')
        for ax in axes:ax.grid(alpha=.18)
        label=self.trace.get('metadata',{}).get('label','One fixed prompt; sequential requests; observed events, no interpolated token arrivals')
        fig.suptitle(label,fontsize=14,fontweight='bold')
        fig.savefig(output/'trajectory.png',dpi=170);fig.savefig(output/'trajectory.svg');plt.close(fig)
        public={'schema_version':1,'metadata':self.trace.get('metadata',{}),'comparisons':stats,
                'completion_definition':'Last observed commit, not final cache synchronization; fixed token budget may end before EOS.',
                'limits':'One selected trajectory cannot establish dataset-wide speedup or timing reproducibility.'}
        (output/'trajectory-stats.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
        examples=self.rejection_examples()
        if examples:self.draw_rejections(examples,output/'rejections.png')
        (output/'rejection-examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
        return public
    def rejection_examples(self):
        # Deterministic: first complete round with zero accepted and first with one.
        key='mtp' if 'mtp' in self.lanes else next((k for k in self.lanes if k!='baseline'),None)
        if key is None:return []
        lane=self.lanes[key];selected=[]
        for wanted in (0,1):
            for event_index,event in enumerate(lane['events']):
                proposals=event.get('proposed_token_ids',[])
                if event['type']=='commit' and len(proposals)==2 and event.get('accepted_count')==wanted:
                    before=event['output_count']-len(event['token_ids'])
                    proposal=lane['events'][event_index-1] if event_index and lane['events'][event_index-1]['type']=='draft' else None
                    texts=proposal.get('token_texts',[]) if proposal else []
                    context=self._context_window(lane,before,proposal,event)
                    selected.append({'lane':key,'round':event.get('round'),'time_seconds':event['t'],
                                     **context,'proposal_token_ids':proposals,'proposal_texts':texts,
                                     'confidence':self._confidence_fields(proposal,event),
                                     'proposal_statuses':['accepted' if i<wanted else 'rejected' if i==wanted else 'discarded_suffix' for i in range(2)],
                                     'accepted_count':wanted,'committed_token_ids':event['token_ids'],
                                     'committed_texts':event.get('token_texts',[]),'output_count_before':before,
                                     'committed_origins':['accepted_draft' if i<wanted else 'target_token' for i in range(len(event['token_ids']))],
                                     'note':'First mismatching proposal is rejected; suffix after it is discarded, not independently accepted/rejected on the true continuation.'})
                    break
        return selected
    @staticmethod
    def _confidence_fields(proposal,event):
        first=(proposal or {}).get('confidence')
        second=event.get('confidence')
        if first is not None and second is not None and first!=second:
            raise ValueError('Draft and commit confidence evidence differs')
        value=second if second is not None else first
        if value is None:return None
        if not isinstance(value,dict) or value.get('decision')!='verify':
            raise ValueError('Rejected proposal requires verified confidence evidence')
        result={key:value.get(key) for key in ('p1','p2','confidence_product','decision')}
        for key in ('p1','p2','confidence_product'):
            number=result[key]
            if type(number) not in (int,float) or not math.isfinite(number) or not 0<=number<=1:
                raise ValueError('Rejected proposal has invalid confidence probability')
        if not math.isclose(result['p1']*result['p2'],result['confidence_product'],rel_tol=1e-10,abs_tol=1e-15):
            raise ValueError('Rejected proposal confidence product disagrees with p1/p2')
        return result

    @staticmethod
    def _rejection_header(item):
        header=f"Round {item['round']} · {item['time_seconds']:.3f} s · accepted {item['accepted_count']}/2 draft tokens"
        confidence=item.get('confidence')
        if confidence:
            header+=(f"\nDraft p1={confidence['p1']:.3g} · p2={confidence['p2']:.3g}"
                     f" · product={confidence['confidence_product']:.3g}")
        return header

    @staticmethod
    def _context_window(lane,before,proposal,event):
        # Use generated-token boundaries, while retaining the proposal's exact
        # stable contextual decode (byte fragments may revise the final glyph).
        pieces=lane.get('token_texts')
        valid_pieces=(isinstance(pieces,list) and len(pieces)==len(lane['token_ids'])
                      and all(isinstance(piece,str) for piece in pieces))
        context=None
        for candidate in (proposal or {},event):
            if isinstance(candidate.get('context_text'),str):
                context=candidate['context_text'];break
        prefixes=lane.get('decoded_prefixes')
        if context is None and isinstance(prefixes,list) and before and len(prefixes)>=before:
            if isinstance(prefixes[before-1],str):context=prefixes[before-1]
        if valid_pieces:
            start=max(0,before-12)
            earlier=''.join(pieces[:start])
            if context is None:context=''.join(pieces[:before])
            if context.startswith(earlier):
                return {'context':context[len(earlier):], 'context_selection':'last_12_generated_token_pieces',
                        'context_token_count':before-start, 'context_token_ids':lane['token_ids'][start:before]}
        return {'context':(context or '')[-160:], 'context_selection':'last_160_characters_fallback',
                'context_token_count':None, 'context_token_ids':None}

    @staticmethod
    def draw_rejections(examples,path):
        if not examples:return None
        import matplotlib.pyplot as plt
        import textwrap
        fig,axes=plt.subplots(len(examples),1,figsize=(13,3.3*len(examples)),squeeze=False,layout='constrained')
        for ax,item in zip(axes[:,0],examples):
            ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
            ax.text(0,.98,TrajectoryReport._rejection_header(item),fontsize=13,fontweight='bold',va='top')
            context='\n'.join(textwrap.wrap(item['context'].replace('\n',' ↵ '),width=125)[-2:])
            ax.text(0,.76,'Previous context: '+context,fontsize=10,color='#55616f',va='top')
            ax.text(0,.47,'Proposed:',fontsize=11)
            for i,token in enumerate(item['proposal_token_ids']):
                text=repr(item['proposal_texts'][i]) if i<len(item['proposal_texts']) else str(token)
                status='ACCEPTED' if i<item['accepted_count'] else 'REJECTED' if i==item['accepted_count'] else 'DISCARDED SUFFIX'
                color='#16845b' if i<item['accepted_count'] else '#c43c50' if i==item['accepted_count'] else '#727c89'
                ax.text(.13+i*.37,.47,f'{text}  [ID {token}]\n{status}',fontfamily='DejaVu Sans Mono',fontsize=11,color=color,bbox={'boxstyle':'round,pad=.5','fc':color+'18','ec':color})
            ax.text(0,.17,'Committed:',fontsize=11)
            for i,token in enumerate(item['committed_token_ids']):
                text=repr(item['committed_texts'][i]) if i<len(item['committed_texts']) else str(token)
                color='#16845b' if i<item['accepted_count'] else '#4263a6'
                origin='accepted draft' if i<item['accepted_count'] else 'target token'
                ax.text(.13+i*.29,.17,f'{text}  [ID {token}]\n{origin}',fontfamily='DejaVu Sans Mono',fontsize=11,color=color,bbox={'boxstyle':'round,pad=.5','fc':color+'18','ec':color})
        fig.suptitle('Actual rejected proposals and the committed correction',fontsize=15,fontweight='bold')
        fig.savefig(path,dpi=170);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);raw=a.trace.read_bytes();trace=json.loads(raw)
    trace.setdefault('metadata',{})['trace_sha256']=sha256(raw).hexdigest()
    # Keep the public plot input metadata portable, not full local model/source paths.
    trace['metadata']={k:v for k,v in trace['metadata'].items() if k in ('dataset_index','label','trace_sha256','sampling','ignore_eos','source','timing_origin')}
    result=TrajectoryReport(trace).write(a.output)
    print(json.dumps({k:{n:v for n,v in x.items() if n!='time_saved_to_each_token_seconds'} for k,x in result['comparisons'].items()},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
