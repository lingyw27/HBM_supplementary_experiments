import ast,json,os,subprocess,textwrap
from pathlib import Path
R=Path('/tf/wjy'); B=R/'results_hbm_int8_rerun_20260916'/'HumanEval_2026091201_worker.py'; O=R/'srlr'/'results_int8_humaneval_lsb_msb'; P=R/'.venv-quant'/'bin'/'python'; S=(2026091201,2026091202,2026091203,2026091204,2026091205); O.mkdir(parents=True,exist_ok=True)
L=r'''
def load_quantized_model(bits):
 import json as J,os as X
 from pathlib import Path as Q
 import numpy as N,torch as T
 from transformers import AutoModelForCausalLM as A
 from safetensors.torch import load_file as F
 from optimum.quanto import requantize as R,quantization_map as M
 global model,config
 assert bits==8
 release_model(); set_seed(42); g=int(X.environ.get('SRLR_LOGICAL_GPU','0'))
 b=Q('/tf/wjy/models/Meta-Llama-3-8B-Instruct'); q=Q('/tf/wjy/models/Meta-Llama-3-8B-Instruct-Quanto-INT8-clean')
 model=A.from_pretrained(str(b),local_files_only=True,torch_dtype=T.float16,device_map={'':'cpu'},attn_implementation='sdpa').eval()
 cfg=J.loads((q/'quantization_map.json').read_text()); state=F(str(q/'model.safetensors'),device='cpu'); R(model,state,cfg,device=T.device('cuda:'+str(g))); config=M(model); assert len(config)==224
 changed=values=0
 for name,info in sorted(config.items()):
  if info.get('weights')!='qint8': continue
  raw=model.get_submodule(name).weight._data; a=raw.detach().cpu().contiguous().numpy().view(N.uint8).reshape(-1); msb=a>>N.uint8(7)
  changed+=int(N.count_nonzero((a&N.uint8(1))!=msb)); values+=int(a.size); a[:]=(a&N.uint8(254))|msb; raw.copy_(T.from_numpy(a.view(N.int8).reshape(tuple(raw.shape))).to(raw.device))
 print('SRLR_ENCODE|changed_lsb={}|values={}|rule=LSB<-MSB'.format(changed,values),flush=True); model=model.eval()
'''
C=r'''
import numpy as _np,torch as _torch
pairs=values=0
for name in sorted(config):
 raw=model.get_submodule(name).weight._data; a=raw.detach().cpu().contiguous().numpy().view(_np.uint8).reshape(-1); bad=(((a>>_np.uint8(7))&_np.uint8(1))^(a&_np.uint8(1)))!=0; n=int(_np.count_nonzero(bad))
 if n: a[bad]=a[bad]|_np.uint8(129); raw.copy_(_torch.from_numpy(a.view(_np.int8).reshape(tuple(raw.shape))).to(raw.device))
 pairs+=n; values+=n
print('SRLR_CORRECT|mismatch_pairs={}|restored_values={}|rule=mismatch->11'.format(pairs,values),flush=True)
'''
def make(seed):
 t=ast.parse(B.read_text(encoding='utf-8').replace('2026091201',str(seed))); lb=ast.parse(textwrap.dedent(L)).body[0].body; cb=ast.parse(textwrap.dedent(C)).body; a=z=False
 for n in ast.walk(t):
  if isinstance(n,ast.FunctionDef) and n.name=='load_quantized_model' and not a: n.body=lb; a=True
  if isinstance(n,ast.FunctionDef) and n.name=='inject_one_to_zero' and not z:
   for i,s in enumerate(n.body):
    if isinstance(s,ast.Return): n.body[i:i]=cb; z=True; break
 assert a and z
 w=O/('HumanEval_{}_worker.py'.format(seed)); w.write_text(ast.unparse(ast.fix_missing_locations(t))+'\n',encoding='utf-8'); return w
def kv(s): return dict(x.split('=',1) for x in s.strip().split('|')[1:] if '=' in x)
with (R/'srlr'/'run_srlr_humaneval_controller.log').open('w',encoding='utf-8',buffering=1) as ctl:
 for i,seed in enumerate(S,1):
  rf=O/('HumanEval_{}.json'.format(seed))
  if rf.exists():
   try:
    if json.loads(rf.read_text()).get('status')=='completed': ctl.write('SKIP HumanEval {}/5 seed={} already completed\n'.format(i,seed)); continue
   except Exception: pass
  w=make(seed); lf=O/('HumanEval_{}.log'.format(seed)); ctl.write('START HumanEval {}/5 seed={}\n'.format(i,seed)); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'1','SRLR_LOGICAL_GPU':'0','SRLR_SEED':str(seed),'SEED':str(seed)})
  with lf.open('w',encoding='utf-8',buffering=1) as log:
   p=subprocess.Popen([str(P),'-u',str(w),str(seed)],cwd=str(R),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); m={}; ref=None
   for line in p.stdout:
    log.write(line); ctl.write(line)
    if line.startswith('ONEZERO_BITS|'): m['onezero_bits']=kv(line)
    elif line.startswith('SRLR_ENCODE|'): m['srlr_encode']=kv(line)
    elif line.startswith('SRLR_CORRECT|'): m['srlr_correct']=kv(line)
    elif line.startswith('RESULT_JSON|'): ref=line.strip().split('|',1)[1]
   rc=p.wait(); wr={}
   if ref:
    try:
     rp=Path(ref)
     if rp.exists(): wr=json.loads(rp.read_text())
    except Exception: pass
  rf.write_text(json.dumps({'status':'completed' if rc==0 else 'failed','task':'HumanEval','seed':seed,'physical_gpu':1,'target_ber':0.003,'protection':'LSB<-MSB; mismatch->11','worker_returncode':rc,'worker_result':wr,'metrics':m},ensure_ascii=False,indent=2),encoding='utf-8'); ctl.write('DONE HumanEval {}/5 seed={} rc={} json={}\n'.format(i,seed,rc,rf))
