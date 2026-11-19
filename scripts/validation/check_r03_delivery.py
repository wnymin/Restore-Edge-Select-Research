from pathlib import Path
import json,re

def main():
 root=Path(__file__).resolve().parents[2]
 required=['docs/r03-met-01.md','docs/r03-inv-01.md','docs/r03-app-01.md','docs/r03-app-02.md','docs/r03-int-01.md','docs/r03-completion.md','docs/evidence/r03-app01-summary.json','docs/evidence/r03-app01-interactions.json','docs/evidence/r03-task-costs.json','docs/evidence/r03-task-costs-holdout.json','docs/evidence/r03-factor-ablations.json','docs/evidence/r03-factor-ablations-summary.json']
 missing=[p for p in required if not (root/p).exists()]
 if missing: raise SystemExit('missing: '+', '.join(missing))
 inv=(root/'docs/r03-inv-01.md').read_text(encoding='utf-8-sig')
 matrix=(root/'docs/r03-int-01.md').read_text(encoding='utf-8-sig')
 for i in range(1,17):
  token=f'| F{i:02d} |'
  if f'F{i:02d}' not in inv or matrix.count(token)!=1: raise SystemExit(f'factor alignment failed: {token}')
 summary=json.loads((root/'docs/evidence/r03-app01-summary.json').read_text(encoding='utf-8-sig'))
 assert summary['synthetic_interventions']['records']==864
 interactions=json.loads((root/'docs/evidence/r03-app01-interactions.json').read_text(encoding='utf-8-sig'))
 assert len(interactions['pairs'])==6
 ablations=json.loads((root/'docs/evidence/r03-factor-ablations-summary.json').read_text(encoding='utf-8-sig'))
 assert {row['mode'] for row in ablations['summary']} == {'omit_overlap','omit_inflight','omit_window','omit_cost','omit_competition'}
 print('R03 delivery verified: artifacts, F01-F16 alignment, evidence summary, interactions, and task costs')
if __name__=='__main__': main()
