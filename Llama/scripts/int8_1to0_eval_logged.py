"""Persist results without changing the original evaluation implementation."""
import argparse,datetime,fcntl,hashlib,json,os,re,runpy
from pathlib import Path
ROOT=Path('/tf/wjy')
RESULTS=ROOT/'int8_1to0_results.json'
ORIGINAL=ROOT/'int8_1to0_eval.py'
PAT=re.compile(r'^(MMLU|MathQA|HumanEval) \| INT8 1->0 \| BER=([0-9.]+).*?round (\d+)/(\d+) \| seed=(\d+) \| changed=(\d+)/(\d+) \|.*? (\d+)/(\d+) \| accuracy: ([0-9.]+)%')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(line, historical=False, provenance=None):
 m=PAT.search(line.strip())
 if not m: return False
 task,ber,trial,rounds,seed,changed,total,correct,n,accuracy=m.groups()
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r=dict(task=task,seed=int(seed),changed_bits=int(changed),total_bits=int(total),actual_ber=int(changed)/int(total),correct=int(correct),total_questions=int(n),accuracy=int(correct)/int(n),quantization_config={'weights':'qint8','activations':'none','label':'Quanto W8A16'},torch_dtype='torch.float16',script_sha256=None if historical else sha(ORIGINAL),logger_sha256=sha(Path(__file__)),timestamp=None if historical else now,recorded_at=now,historical=historical,provenance=provenance,raw_output=line.strip(),metadata_note='Historical execution timestamp and script hash unavailable; configuration inferred from associated notebook code.' if historical else 'Original evaluator hash captured at result emission.')
 with (ROOT/'int8_1to0_results.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rows=json.loads(RESULTS.read_text()) if RESULTS.exists() else []
  if historical and any(x.get('task')==task and x.get('seed')==int(seed) and x.get('raw_output')==line.strip() for x in rows): return False
  rows.append(r)
  tmp=RESULTS.with_suffix('.json.tmp')
  tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
  os.replace(tmp,RESULTS)
 return True
def backfill():
 count=0
 for p in sorted(ROOT.glob('*.ipynb')):
  nb=json.loads(p.read_text())
  for i,c in enumerate(nb.get('cells',[])):
   for o in c.get('outputs',[]):
    t=o.get('text',o.get('data',{}).get('text/plain',''))
    t=''.join(t) if isinstance(t,list) else str(t)
    for line in t.splitlines():
     if record(line,True,dict(notebook=str(p),cell_index=i,execution_count=c.get('execution_count'))): count+=1
 print('BACKFILLED',count,'FILE',RESULTS)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--backfill',action='store_true')
 ap.add_argument('--notebook',choices=['MMUL.ipynb','mathQA.ipynb','humaneval.ipynb'])
 ap.add_argument('--gpu',default='0')
 args=ap.parse_args()
 if args.backfill: backfill(); return
 assert args.notebook, '--notebook required'
 ns=runpy.run_path(str(ORIGINAL),run_name='original_int8_evaluator')
 import builtins
 def logged_print(*items,**kw):
  builtins.print(*items,**kw)
  line=kw.get('sep',' ').join(map(str,items))
  if record(line,False,dict(notebook=args.notebook,gpu=args.gpu)):
   builtins.print('JSON_SAVED',str(RESULTS),flush=True)
 ns['run'].__globals__['print']=logged_print
 ns['run'](args.notebook,args.gpu)
if __name__=='__main__': main()
