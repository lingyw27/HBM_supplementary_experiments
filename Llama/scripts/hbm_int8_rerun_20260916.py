import ast,datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/tf/wjy')
OUT=ROOT/'results_hbm_int8_rerun_20260916'
SEEDS=(2026091201,2026091202,2026091203,2026091204,2026091205)
TASKS={'MMUL.ipynb':'MMLU','mathQA.ipynb':'MathQA','humaneval.ipynb':'HumanEval'}
PYTHON=ROOT/'.venv-quant/bin/python'
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def build(notebook):
 raw=(ROOT/notebook).read_text(); found=[]
 for c in json.loads(raw)['cells']:
  if c.get('cell_type')!='code': continue
  try: t=ast.parse(''.join(c.get('source',[])))
  except SyntaxError: continue
  for n in t.body:
   if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='quant_code' for x in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
    inner=ast.parse(n.value.value)
    if 'load_quantized_model' in {x.name for x in inner.body if isinstance(x,ast.FunctionDef)} and TASKS[notebook]+' |' in n.value.value: found.append(inner)
 assert len(found)==1,(notebook,len(found))
 t=found[0]; assert isinstance(t.body[-1],ast.Try); t.body.pop()
 original_base=ast.unparse(t)
 assert 'torch.float16' in original_base and 'torch.bfloat16' not in original_base
 assert 'quantize(model' in original_base and 'qint8' in original_base
 for fn in t.body:
  if isinstance(fn,ast.FunctionDef) and fn.name in ('run_evaluation','run_humaneval'):
   for statement in fn.body:
    if isinstance(statement,ast.For) and isinstance(statement.target,ast.Name) and statement.target.id=='start':
     statement.body.append(ast.parse("print('PROGRESS', start+len(rows), '/', len(test), 'correct=', correct, flush=True)").body[0])
 ast.fix_missing_locations(t)
 s=(ROOT/'int8_1to0_eval.py').read_text(); a=ast.parse(s)
 fault=next(ast.literal_eval(n.value) for n in a.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FAULT_CODE' for x in n.targets))
 fault=fault.replace('original_loader(bits)',"print('LOAD original FP16 Quanto INT8', flush=True)\n    original_loader(bits)\n    cfg = quantization_map(model)\n    expected = {f'model.layers.{i}.{part}' for i in range(32) for part in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')}\n    assert set(cfg) == expected\n    assert all(p.dtype == torch.float16 for p in model.parameters() if p.is_floating_point())\n    print('INJECT START: 224 matrices, no lm_head or embeddings', flush=True)")
 fault=fault.replace('chosen = uniform_subset(total_ones, k, seed)',"print('BITS BEFORE', total_bits, 'ONES', total_ones, 'K', k, flush=True)\n    chosen = uniform_subset(total_ones, k, seed)\n    print('SAMPLING DONE', flush=True)")
 fault=fault.replace('verified += changed',"verified += changed\n            print('INJECT VERIFIED', verified, '/', k, name, flush=True)")
 entry="""
import json,datetime,time
print('WORKER PID',os.getpid(),'GPU',os.environ['CUDA_VISIBLE_DEVICES'],'SEED',seed,flush=True)
try:
    load_quantized_model(8)
    print('EVAL START',flush=True)
    correct,total=run_evaluation()
    print('RESULT_JSON|'+json.dumps({'correct':correct,'total_questions':total,'accuracy':correct/total}),flush=True)
finally:
    release_model()
"""
 program=ast.unparse(t)+'\n'+fault+'\n'+entry
 compile(program,'<rerun>','exec')
 return program,dict(notebook_sha256=digest(raw),original_evaluation_sha256=digest(original_base),injection_script_sha256=digest(s),script_sha256=digest(program))
def acquire_gpu():
 while True:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  for gpu,free in sorted([tuple(map(int,line.split(','))) for line in raw.strip().splitlines()],key=lambda x:-x[1]):
   if free<24576: continue
   lock=(ROOT/'.tmp'/f'int8_1to0_gpu{gpu}.lock').open('a')
   try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError: lock.close(); continue
   print('GPU SELECTED',gpu,'free MiB',free,flush=True); return gpu,lock
  print('WAITING FOR GPU >=24GiB FREE',flush=True); time.sleep(60)
def main():
 OUT.mkdir(exist_ok=True); (ROOT/'.tmp').mkdir(exist_ok=True)
 campaign=(OUT/'campaign.lock').open('a'); fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prepared={n:build(n) for n in TASKS}
 for notebook,label in TASKS.items():
  program,meta=prepared[notebook]
  for trial,seed in enumerate(SEEDS,1):
   resultpath=OUT/f'{label}_{seed}.json'
   if resultpath.exists() and json.loads(resultpath.read_text()).get('status')=='completed': continue
   gpu,lock=acquire_gpu()
   logpath=OUT/f'{label}_{seed}.log'
   workerpath=OUT/f'{label}_{seed}_worker.py'; workerpath.write_text(program)
   record=dict(task=label,seed=seed,round=trial,torch_dtype='torch.float16',quantization_config={'weights':'qint8','activations':'none','granularity':'per_output_channel','matrices':224},timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,status='running',**meta)
   resultpath.write_text(json.dumps(record,indent=2))
   env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
   env['PATH']=str(PYTHON.parent)+os.pathsep+env.get('PATH','')
   for key,folder in [('TMPDIR','.tmp'),('HF_HOME','.cache/huggingface'),('TORCH_HOME','.cache/torch'),('TORCH_EXTENSIONS_DIR','.cache/torch_extensions')]:
    d=ROOT/folder; d.mkdir(parents=True,exist_ok=True); env[key]=str(d)
   print('START',label,trial,'/5',seed,flush=True)
   result=None; bits=None
   try:
    with logpath.open('w',buffering=1) as log:
     p=subprocess.Popen([str(PYTHON),'-u',str(workerpath),str(seed)],cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
     try:
      for line in p.stdout:
       print(line,end='',flush=True); log.write(line)
       if line.startswith('ONEZERO_BITS|'): bits=tuple(map(int,line.strip().split('|')[1].split('/')))
       if line.startswith('RESULT_JSON|'): result=json.loads(line.split('|',1)[1])
      rc=p.wait()
     except BaseException:
      if p.poll() is None:
       p.terminate()
       try: p.wait(timeout=10)
       except subprocess.TimeoutExpired: p.kill(); p.wait()
      raise
     finally: p.stdout.close()
    if bits: record.update(changed_bits=bits[0],total_bits=bits[1],actual_ber=bits[0]/bits[1])
    if rc==0 and result is not None and bits is not None: record.update(result,status='completed')
    else: record.update(status='failed',exit_code=rc)
   except BaseException as e:
    record.update(status='interrupted',error=repr(e)); raise
   finally:
    record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    resultpath.write_text(json.dumps(record,indent=2)); lock.close()
   print('ROUND_STATUS',json.dumps(record),flush=True)
   if record['status']!='completed': raise RuntimeError(f'Failed round; inspect {logpath}')
 print('ALL_15_COMPLETED',flush=True)
if __name__=='__main__': main()
