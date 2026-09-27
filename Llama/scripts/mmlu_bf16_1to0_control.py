import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import ast,json,hashlib,datetime,time,fcntl
from pathlib import Path
ROOT=Path('/tf/wjy')
nb=json.loads((ROOT/'MMUL.ipynb').read_text())
candidates=[]
for c in nb['cells']:
 try: t=ast.parse(''.join(c.get('source',[])))
 except SyntaxError: continue
 for n in t.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='quant_code' for x in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
   inner=ast.parse(n.value.value)
   if 'load_quantized_model' in {x.name for x in inner.body if isinstance(x,ast.FunctionDef)} and 'MMLU |' in n.value.value: candidates.append(inner)
assert len(candidates)==1
tree=candidates[0]
assert isinstance(tree.body[-1],ast.Try)
tree.body.pop()
class BF16(ast.NodeTransformer):
 def visit_Attribute(self,n):
  if isinstance(n.value,ast.Name) and n.value.id=='torch' and n.attr=='float16': n.attr='bfloat16'
  return self.generic_visit(n)
tree=BF16().visit(tree)
ast.fix_missing_locations(tree)
base=ast.unparse(tree)
s=ast.parse((ROOT/'int8_1to0_eval.py').read_text())
fault=next(ast.literal_eval(n.value) for n in s.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FAULT_CODE' for x in n.targets))
fault=fault.split('import sys')[0]
compile(base+'\n'+fault,'<bf16_control>','exec')
report=dict(task='MMLU',seed=2026091201,torch_dtype='torch.bfloat16',quantization_config={'weights':'qint8','activations':'none','granularity':'per_output_channel'},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),generated_sha256=hashlib.sha256((base+'\n'+fault).encode()).hexdigest(),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
lock=(ROOT/'.tmp/int8_1to0_gpu0.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print('BF16 START',report,flush=True)
exec(base+'\n'+fault,globals())
hooks=[]; flags={}
def tensors(x):
 if torch.is_tensor(x): yield x
 elif isinstance(x,dict):
  for v in x.values(): yield from tensors(v)
 elif isinstance(x,(tuple,list)):
  for v in x: yield from tensors(v)
def monitor(name):
 def hook(module,args,out):
  for t in tensors(out):
   if not t.is_floating_point(): continue
   v=torch.stack((torch.isinf(t).any(),torch.isnan(t).any()))
   flags[name]=v if name not in flags else flags[name]|v
 return hook
try:
 print('LOAD BF16 original weights + Quanto qint8',flush=True)
 load_quantized_model(8)
 print('INJECT START',flush=True)
 changed,total=inject_one_to_zero(2026091201)
 report.update(changed_bits=changed,total_bits=total,actual_ber=changed/total)
 over=nonfinite=0; peak=0.0
 for name in quantization_map(model):
  w=model.get_submodule(name).weight
  raw=w._data; scale=w._scale
  for start in range(0,raw.shape[0],128):
   effective=raw[start:start+128].float()*scale[start:start+128].float()
   over+=int((effective.abs()>65504).sum().item())
   nonfinite+=int((~torch.isfinite(effective)).sum().item())
   peak=max(peak,float(effective.abs().max().item()))
 report.update(weights_abs_gt_fp16_max=over,nonfinite_weights=nonfinite,max_abs_weight=peak)
 print('WEIGHT_STATS',report,flush=True)
 for name,module in model.named_modules(): hooks.append(module.register_forward_hook(monitor(name)))
 original_generate=generate_batch
 batches=0
 def generate_batch(*args,**kwargs):
  global batches
  result=original_generate(*args,**kwargs); batches+=1
  if batches%10==0: print('BATCHES_COMPLETED',batches,flush=True)
  return result
 print('EVAL START',flush=True)
 correct,n=run_evaluation()
 report.update(correct=correct,total_questions=n,accuracy=correct/n,status='completed')
except BaseException as e:
 report.update(status='failed',error=repr(e))
 raise
finally:
 bad={}
 for name,v in flags.items():
  inf,nan=v.tolist()
  if inf or nan: bad[name]={'inf':inf,'nan':nan}
 report['nonfinite_module_outputs']=bad
 report['monitored_modules']=len(flags)
 if bad and report.get('status')=='completed': report['status']='completed_with_nonfinite_outputs'
 for h in hooks: h.remove()
 (ROOT/'mmlu_bf16_1to0_control_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print('BF16_FINAL',json.dumps(report,ensure_ascii=False),flush=True)
 release_model()
 lock.close()
