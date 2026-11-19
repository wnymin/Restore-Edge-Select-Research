"""Factor-specific closed-loop ablations on the pinned CPU synthetic fixture."""
from __future__ import annotations
import argparse,csv,json
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from scripts.cpu.replay.study_bailian import Experiment,ROOT,read
from restore_edge_select.dynamic_cpu import projected_cost

MODES=('full','omit_overlap','omit_inflight','omit_window','omit_cost','omit_competition')

def score(mode,row):
 r=dict(row)
 if mode in {'omit_overlap','omit_window','omit_competition'}:
  r['head_slack_ms']=0; r['committed_ahead_compute_ms']=0
 if mode in {'omit_inflight','omit_competition'}: r['inflight_ms']=0
 if mode=='omit_competition': r['queue_debt_ms']=0
 if mode=='omit_cost':
  r['incremental_ram_ms']=0; r['incremental_ssd_ms']=0; r['serial_io_ms']=0
 return projected_cost(r)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
 fixture=read(ROOT/'fixtures/phase1-hand-trace.json'); model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
 records=[]
 for shared,rate,budget,arrival in product((False,True),(1,4,16),(0,70,160,192),(1,2,15,29,35,60)):
  trace=deepcopy(fixture); trace.pop('expected_by_prefetch_mib',None)
  if shared: trace['requests'][2]=dict(deepcopy(trace['requests'][1]),id='C')
  trace['cost']['device_copy']={'startup_ms':'0','bytes_per_ms':100_000_000}
  trace['requests'][2]['arrival_ms']=arrival
  trace['cost']['ssd_to_device']['bytes_per_ms']=str(F(trace['cost']['ssd_to_device']['bytes_per_ms'])/rate)
  exp=Experiment(trace,model,budget)
  runs={m:exp.run('snapshot-score',score_candidate=(projected_cost if m=='full' else lambda row,m=m:score(m,row))) for m in MODES}
  full=runs['full']
  for mode,run in runs.items():
   delta=[float(a-b) for a,b in zip(run['completion_ms'],full['completion_ms'])]
   records.append({'sharing':int(shared),'ssd_scale':rate,'budget_mib':budget,'c_arrival_ms':arrival,'mode':mode,'full_mean_ms':float(full['response_mean_ms']),'ablated_mean_ms':float(run['response_mean_ms']),'loss_ms':float(run['response_mean_ms']-full['response_mean_ms']),'decision_changed':int(run['boundaries_tokens']!=full['boundaries_tokens']),'changed_requests':sum(a!=b for a,b in zip(run['boundaries_tokens'],full['boundaries_tokens'])),'max_request_loss_ms':max([0,*delta]),'completion_delta_ms':delta})
 args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'R03-factor-ablation-v1','scope':'CPU synthetic fixture; same input and execution semantics','modes':MODES,'records':records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with args.out.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
  fields=['sharing','ssd_scale','budget_mib','c_arrival_ms','mode','full_mean_ms','ablated_mean_ms','loss_ms','decision_changed','changed_requests','max_request_loss_ms']; w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader(); [w.writerow({k:r[k] for k in fields}) for r in records]
 print(f'wrote {len(records)} records')
if __name__=='__main__':main()
