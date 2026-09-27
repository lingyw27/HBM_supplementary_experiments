from pathlib import Path
import os, sys, json, time, math, hashlib, gc, re, ast, subprocess, shutil, uuid, traceback
ROOT = Path(__file__).resolve().parent
assert Path.cwd().resolve() == ROOT
sys.dont_write_bytecode = True
def safe(path):
    p = (ROOT/path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError(f'路径越界: {p}')
    return p
CFG = json.loads(Path(sys.argv[1]).read_text())
OUT = safe('results_int8')/CFG['result_name']
OUT = safe(OUT); OUT.mkdir(parents=True, exist_ok=True)
def dump(path, obj):
    p = safe(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.writing')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(p)
def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def status(stage, **kw):
    record = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)
    dump(OUT/'status.json', record); print(record, flush=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import t as student_t
import pandas as pd
from collections import defaultdict

def load_data():
    names = ['mathqa_fixed_1000','mathqa_fixed_5shot','mmlu_fixed_1000','mmlu_fixed_5shot','humaneval']
    data = {n:list(load_from_disk(str(safe(n)))) for n in names}
    for name, count in [('mathqa_fixed_1000',1000),('mathqa_fixed_5shot',5),('mmlu_fixed_1000',1000),('humaneval',164)]:
        assert len(data[name]) == count, (name,len(data[name]))
    norm = lambda s: ' '.join(s.lower().split())
    for test, shot, field in [('mathqa_fixed_1000','mathqa_fixed_5shot','Problem'),('mmlu_fixed_1000','mmlu_fixed_5shot','question')]:
        test_keys = [norm(r[field]) for r in data[test]]
        assert len(set(test_keys)) == len(test_keys), f'{test} 题目重复，请确认数据'
        assert not set(test_keys) & {norm(r[field]) for r in data[shot]}, f'{test} 与5-shot存在题目交集'
    by_subject = defaultdict(list)
    for r in data['mmlu_fixed_5shot']: by_subject[r['subject']].append(r)
    for r in data['mmlu_fixed_1000']:
        assert len(by_subject[r['subject']]) == 5, ('MMLU 5-shot',r['subject'])
        assert len(r['choices']) == 4 and 0 <= int(r['answer']) < 4
    assert len({r['task_id'] for r in data['humaneval']}) == 164
    for r in data['humaneval']: assert r['entry_point'].isidentifier()
    dump(OUT/'data_manifest.json', {k:dict(count=len(v),sha256=digest(v)) for k,v in data.items()})
    return data, by_subject

def math_question(r):
    text = r['options']
    try:
        value = ast.literal_eval(text)
    except (ValueError,SyntaxError): value = text
    if isinstance(value,list): text = '\n'.join(value)
    # 使用数据原始选项文字，只规范化 a) -> A)，不按逗号切割选项。
    text = re.sub(r'(?<!\w)([a-e])\s*\)', lambda m:m.group(1).upper()+')', text)
    return r['Problem'].strip()+'\n'+text.strip()

def mmlu_question(r):
    return r['question'].strip()+'\n'+'\n'.join(f'{chr(65+i)}) {s}' for i,s in enumerate(r['choices']))

def build_prompts(tokenizer, data, by_subject):
    prompts = {}
    for key, file in [('mathqa','mathqa_fixed_1000'),('mmlu','mmlu_fixed_1000')]:
        rows=[]
        for i,r in enumerate(data[file]):
            shots = data['mathqa_fixed_5shot'] if key=='mathqa' else by_subject[r['subject']]
            question = math_question if key=='mathqa' else mmlu_question
            answer = (lambda x:x['correct'].strip().upper()) if key=='mathqa' else (lambda x:chr(65+int(x['answer'])))
            messages=[dict(role='system',content='Choose the correct option. Reply with only the uppercase option letter.')]
            for s in shots:
                assert answer(s) in ('ABCDE' if key=='mathqa' else 'ABCD')
                messages += [dict(role='user',content=question(s)),dict(role='assistant',content=answer(s))]
            messages.append(dict(role='user',content=question(r)))
            prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert answer(r) in ('ABCDE' if key=='mathqa' else 'ABCD')
            rows.append(dict(id=i,prompt=prompt,gold=answer(r),labels=list('ABCDE' if key=='mathqa' else 'ABCD'),subject=r.get('subject')))
        prompts[key]=rows
    prompts['humaneval']=[]
    for r in data['humaneval']:
        messages=[dict(role='system',content='Return only Python code, with no explanation. Complete the function and include its full definition. Preserve the signature.'),dict(role='user',content=r['prompt'])]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompts['humaneval'].append(dict(id=r['task_id'],prompt=prompt,problem=r))
    for key,rows in prompts.items():
        lengths=[len(tokenizer.encode(r['prompt'],add_special_tokens=False)) for r in rows]
        reserve = CFG['max_new_tokens'] if key=='humaneval' else 8
        assert max(lengths)+reserve <= CFG['context_limit'], (key,max(lengths),'提高 context_limit，不得静默截断5-shot')
    dump(OUT/'prompt_manifest.json',{k:dict(count=len(v),sha256=digest(v)) for k,v in prompts.items()})
    return prompts



@torch.no_grad()
def quantize_groupwise_rtn(weight, group_size=128):
    assert weight.ndim==2 and group_size==128
    out, ins = weight.shape
    w=weight.float()
    assert torch.isfinite(w).all(), '原始权重非有限'
    groups=(ins+group_size-1)//group_size
    padded=F.pad(w,(0,groups*group_size-ins)).reshape(out,groups,group_size)
    maximum=padded.abs().amax(-1)
    scale=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    q=(padded/scale.unsqueeze(-1)).round().clamp(-127,127).to(torch.int8)
    return q.reshape(out,-1)[:,:ins].contiguous(),scale.contiguous()

class QuantLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.in_features=linear.in_features; self.out_features=linear.out_features
        self.group_size=128
        q,s=quantize_groupwise_rtn(linear.weight)
        self.register_buffer('qweight',q)
        self.register_buffer('scale',s)
        self.register_buffer('bias',None if linear.bias is None else linear.bias.detach().to(torch.bfloat16).clone())
        self.register_buffer('_cache',None,persistent=False)
        self.clean=q.cpu().clone()
        self.scale_clean=s.cpu().clone()
    @torch.no_grad()
    def dequantize_weight(self):
        ins=self.in_features; groups=self.scale.shape[1]
        q=F.pad(self.qweight,(0,groups*128-ins)).reshape(self.out_features,groups,128)
        return (q.float()*self.scale.unsqueeze(-1)).reshape(self.out_features,-1)[:,:ins].to(torch.bfloat16).contiguous()
    @torch.no_grad()
    def refresh(self):
        self._cache=None
        if CFG['cache_dequant']: self._cache=self.dequantize_weight()
    def forward(self,x):
        x=x.to(torch.bfloat16)
        w=self._cache if self._cache is not None else self.dequantize_weight()
        return F.linear(x,w,self.bias)

@torch.no_grad()
def replace_linear_with_quantlinear(model):
    report=[]
    targets=[(name,m) for name,m in model.named_modules() if isinstance(m,nn.Linear)]
    for name,m in targets:
        q=QuantLinear(m)
        # 只对一个小切片做数值校验，避免为整个模型另存 FP32 权重。
        reference=m.weight[:8].float()
        reconstruction=q.dequantize_weight()[:8].float()
        report.append(dict(name=name,shape=list(q.qweight.shape),mse=float((reference-reconstruction).square().mean()),max_abs_error=float((reference-reconstruction).abs().max())))
        parent_name,_,leaf=name.rpartition('.')
        parent=model.get_submodule(parent_name) if parent_name else model
        setattr(parent,leaf,q)
    assert report and not any(isinstance(m,nn.Linear) for m in model.modules())
    dump(OUT/'quantization.json',report)
    return [(n,m) for n,m in model.named_modules() if isinstance(m,QuantLinear)]

@torch.no_grad()
def restore_clean_qweights(layers):
    for name,m in layers:
        m.qweight.copy_(m.clean.to(m.qweight.device))
        assert torch.equal(m.scale.cpu(),m.scale_clean),f'scale 被修改: {name}'
        m.refresh()

@torch.no_grad()
def quantization_self_test():
    linear=nn.Linear(259,5,bias=True,dtype=torch.bfloat16)
    linear.weight[0].zero_()
    q=QuantLinear(linear)
    assert q.qweight.dtype==torch.int8 and q.scale.shape==(5,3)
    assert (q.qweight[0]==0).all() and torch.isfinite(q.scale).all()
    x=torch.randn(3,259,dtype=torch.bfloat16)
    expected=F.linear(x,q.dequantize_weight(),q.bias)
    assert torch.equal(q(x),expected) and q(x).dtype==torch.bfloat16
    clean=q.qweight.clone(); scales=q.scale.clone()
    q.qweight.view(torch.uint8)[1,0]^=128
    assert not torch.equal(clean,q.qweight)
    q.refresh()
    assert torch.equal(q(x),F.linear(x,q.dequantize_weight(),q.bias))
    q.qweight.copy_(clean); q.refresh()
    assert torch.equal(q(x),expected) and torch.equal(q.scale,scales)
    raw=np.array([0,127,-128,-1],dtype=np.int8)
    flipped=np.bitwise_xor(raw.view(np.uint8),np.uint8(128)).view(np.int8)
    assert flipped.tolist()==[-128,-1,0,127]
    print('PASS: group128尾组/零组、BF16 forward、符号位、缓存刷新和恢复',flush=True)



def make_fault_masks(layers, seed, folder, ber):
    folder=safe(folder); folder.mkdir(parents=True,exist_ok=True)
    sizes=np.array([m.qweight.numel() for _,m in layers],dtype=np.int64)
    boundaries=np.cumsum(sizes)*8
    total=int(boundaries[-1]); target=int(round(total*ber))
    assert 0<=ber<=1 and total>0
    rng=np.random.default_rng(seed)
    masks=[]; selected=0; block=262144
    for j,((name,m),size) in enumerate(zip(layers,sizes)):
        path=folder/f'{j:04d}.mask'
        mask=np.memmap(path,mode='w+',dtype=np.uint8,shape=(int(size),))
        for start in range(0,int(size),block):
            n=min(block,int(size)-start); count=int(rng.binomial(n*8,ber))
            ids=rng.choice(n*8,count,replace=False)
            values=np.zeros(n,dtype=np.uint8)
            np.bitwise_or.at(values,ids//8,(1<<(ids%8)).astype(np.uint8))
            mask[start:start+n]=values; selected+=count
        masks.append(mask)
    # 均匀地从已选位中移除，或从未选位中补充；同一批先去重。
    delta=target-selected
    while delta:
        want_on=delta<0
        acceptance=ber if want_on else 1-ber
        count=min(2000000,max(1024,int(abs(delta)/max(acceptance,1e-9)*1.3)))
        candidates=np.unique(rng.integers(0,total,size=count,dtype=np.int64))
        rng.shuffle(candidates)
        for g in candidates:
            layer=int(np.searchsorted(boundaries,g,side='right'))
            local=int(g-(int(boundaries[layer-1]) if layer else 0))
            index,bit=divmod(local,8); flag=np.uint8(1<<bit)
            is_on=bool(masks[layer][index]&flag)
            if is_on==want_on:
                masks[layer][index]^=flag
                delta += 1 if want_on else -1
                if delta==0: break
    for mask in masks: mask.flush()
    return masks,total,target


def fault_self_test():
    layer=QuantLinear(nn.Linear(259,17,dtype=torch.bfloat16)); layers=[('toy',layer)]
    before=layer.qweight.clone()
    folder=OUT/'self_tests'/uuid.uuid4().hex
    info=inject_random_bit_errors(layers,5,folder/'a',0.003)
    corrupted=layer.qweight.clone()
    assert info['actual_flips']==round(before.numel()*8*0.003)
    again=inject_random_bit_errors(layers,5,folder/'b',0.003)
    assert torch.equal(corrupted,layer.qweight), '固定seed不可复现'
    restore_clean_qweights(layers)
    assert torch.equal(before,layer.qweight),'恢复失败'
    zero=inject_random_bit_errors(layers,6,folder/'zero',0)
    assert zero['actual_flips']==0 and torch.equal(before,layer.qweight)
    print('PASS: 精确BER、无重复bit、固定seed复现、零BER、逐轮恢复',flush=True)



# 子进程程序：在执行测试和生成代码之前安装内核级限制。
SANDBOX_RUNNER = r"""
import os, sys, json, ctypes, ctypes.util, resource, signal
from pathlib import Path
job=json.loads(Path(sys.argv[1]).read_text())
work=Path.cwd().resolve()
libc=ctypes.CDLL(None,use_errno=True)
libc.syscall.restype=ctypes.c_long
class Ruleset(ctypes.Structure): _fields_=[('handled_access_fs',ctypes.c_uint64)]
class Beneath(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
def fail(msg):
    print(json.dumps(dict(infrastructure_error=msg)),flush=True); sys.exit(70)
try:
    abi=libc.syscall(444,0,0,1)
    if abi<1: raise RuntimeError('Landlock不可用')
    rights=sum(1<<i for i in [1,4,5,6,7,8,9,10,11,12])
    if abi>=2: rights|=1<<13
    if abi>=3: rights|=1<<14
    rules=Ruleset(rights)
    fd=libc.syscall(444,ctypes.byref(rules),ctypes.sizeof(rules),0)
    if fd<0: raise OSError(ctypes.get_errno(),'create_ruleset')
    pfd=os.open(str(work),os.O_PATH|os.O_CLOEXEC)
    rule=Beneath(rights,pfd)
    if libc.syscall(445,fd,1,ctypes.byref(rule),0)<0: raise OSError(ctypes.get_errno(),'add_rule')
    if libc.prctl(38,1,0,0,0): raise OSError('no_new_privs')
    if libc.syscall(446,fd,0)<0: raise OSError(ctypes.get_errno(),'restrict_self')
    os.close(fd); os.close(pfd)
    sec=ctypes.CDLL(ctypes.util.find_library('seccomp') or 'libseccomp.so.2',use_errno=True)
    sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]; sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)
    if not ctx: raise RuntimeError('seccomp_init failed')
    blocked=['socket','socketpair','connect','bind','listen','accept','accept4','sendto','sendmsg','sendmmsg','recvmsg','recvmmsg','clone','clone3','fork','vfork','execve','execveat','ptrace','mount','umount2','pivot_root','chroot','unshare','setns','kill','tkill','tgkill','bpf','process_vm_writev','process_vm_readv','open_by_handle_at','io_uring_setup']
    for name in blocked:
        nr=sec.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and sec.seccomp_rule_add(ctx,0x00050000|1,nr,0)!=0: raise RuntimeError(name)
    if sec.seccomp_load(ctx)!=0: raise RuntimeError('seccomp_load failed')
    sec.seccomp_release(ctx)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(job['timeout'],job['timeout']+1))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
except BaseException as e: fail(type(e).__name__+': '+str(e))
# 无论候选代码如何输出，仅父进程读取结果文件；正式结果在沙箱外。
result_path=Path('test_result.json')
try:
    namespace={'__name__':'__main__'}
    exec(compile(job['code']+'\n'+job['test']+'\ncheck('+job['entry_point']+')','candidate.py','exec'),namespace)
    result=dict(passed=True,error=None)
except BaseException as e:
    result=dict(passed=False,error=type(e).__name__+': '+str(e)[:1500])
result_path.write_text(json.dumps(result))
"""

def check_solution(code,problem,folder):
    folder=safe(folder); folder.mkdir(parents=True,exist_ok=False)
    runner=folder/'runner.py'; runner.write_text(SANDBOX_RUNNER)
    job=folder/'job.json'
    dump(job,dict(code=code,test=problem['test'],entry_point=problem['entry_point'],timeout=CFG['humaneval_timeout']))
    env={k:v for k,v in os.environ.items() if k in ['PATH','LANG','LC_ALL','LD_LIBRARY_PATH']}
    env.update(HOME=str(folder),TMPDIR=str(folder),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    log=folder/'process.log'
    try:
        with log.open('w') as f:
            process=subprocess.run([sys.executable,'-I','-B',str(runner),str(job)],cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=CFG['humaneval_timeout']+5)
    except subprocess.TimeoutExpired:
        return dict(passed=False,error='wall-clock timeout')
    if process.returncode==70:
        raise RuntimeError('HumanEval隔离不可用: '+log.read_text()[-3000:])
    result=folder/'test_result.json'
    if not result.exists(): return dict(passed=False,error=f'process exit {process.returncode}; '+log.read_text()[-1000:])
    return json.loads(result.read_text())

def sandbox_self_test():
    base=OUT/'sandbox_self_tests'/uuid.uuid4().hex
    # 禁止写入的路径也在 lyw_new 内，绝不尝试改动其他文件夹。
    readonly=safe(base/'not_writable.txt'); readonly.parent.mkdir(parents=True,exist_ok=True); readonly.write_text('keep')
    code='def probe():\n    import socket\n    try:\n        open('+repr(str(readonly))+', "w").write("bad")\n        return False\n    except PermissionError:\n        pass\n    try:\n        socket.socket()\n        return False\n    except PermissionError:\n        return True\n'
    problem=dict(test='def check(candidate):\n    assert candidate() is True',entry_point='probe')
    result=check_solution(code,problem,base/'probe')
    assert result['passed'] and readonly.read_text()=='keep',result
    bad=check_solution('def probe(): return False',problem,base/'bad')
    assert not bad['passed'],'测试失败没有被识别'
    print('PASS: Landlock写入限制、seccomp网络限制、通过/失败判定',flush=True)

def extract_code(raw,problem):
    # 只做固定格式处理；不根据测试结果挑候选、不回填 canonical_solution。
    text=raw.split('</think>')[-1].strip()
    match=re.search(r'\x60\x60\x60(?:python)?\s*\n(.*?)\x60\x60\x60',text,re.S|re.I)
    if match: text=match.group(1)
    if re.search(r'^\s*(?:async\s+)?def\s+'+re.escape(problem['entry_point'])+r'\s*\(',text,re.M):
        return text
    # 某些模型仅返回函数体，保留其原始缩进。
    body=raw if not match else match.group(1)
    if body.startswith(('    ','\t')): return problem['prompt']+body
    return text



def read_jsonl(path):
    if not path.exists(): return []
    rows=[]
    lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1: raise
            # 中断造成尾行不完整：完整记录保存在原文件备份中，再原子修复。
            backup=path.with_name(path.name+'.partial_'+uuid.uuid4().hex)
            shutil.copy2(path,backup)
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return rows

@torch.inference_mode()
def choice_scores(model,tokenizer,row):
    prefix=tokenizer.encode(row['prompt'],add_special_tokens=False)
    scores=[]
    # 单次prefill + KV复用，支持候选为多token；答案不参与prompt。
    ids=torch.tensor([prefix],device=model.device)
    output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1)
    if not torch.isfinite(output.logits).all(): return None
    first=output.logits[0,-1].float().log_softmax(-1)
    for label in row['labels']:
        continuation=tokenizer.encode(label,add_special_tokens=False)
        assert continuation
        if len(continuation)==1:
            scores.append(float(first[continuation[0]])); continue
        full=prefix+continuation
        inp=torch.tensor([full],device=model.device)
        logits=model(input_ids=inp,attention_mask=torch.ones_like(inp),use_cache=False,logits_to_keep=len(continuation)+1).logits[0].float()
        if not torch.isfinite(logits).all(): return None
        token_logits=logits[-len(continuation)-1:-1].log_softmax(-1)
        scores.append(float(token_logits[torch.arange(len(continuation),device=logits.device),torch.tensor(continuation,device=logits.device)].sum()))
    return scores

@torch.inference_mode()
def evaluate_dataset(model,tokenizer,key,rows,run_dir):
    path=run_dir/(key+'.jsonl'); previous=read_jsonl(path)
    done={str(r['id']):r for r in previous}
    assert len(done)==len(previous) and set(done)<={str(r['id']) for r in rows}
    with path.open('a',buffering=1) as stream:
        for i,row in enumerate(rows):
            if str(row['id']) in done: continue
            start=time.monotonic()
            if key!='humaneval':
                scores=choice_scores(model,tokenizer,row)
                pred=None if scores is None else row['labels'][int(np.argmax(scores))]
                result=dict(id=row['id'],prediction=pred,gold=row['gold'],correct=pred==row['gold'],scores=scores,subject=row['subject'],error='nonfinite logits' if scores is None else None)
            else:
                tokens=tokenizer(row['prompt'],return_tensors='pt',add_special_tokens=False).to(model.device)
                generated=model.generate(**tokens,do_sample=False,num_beams=1,max_new_tokens=CFG['max_new_tokens'],pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
                new=generated[0,tokens.input_ids.shape[1]:]
                raw=tokenizer.decode(new,skip_special_tokens=False)
                readable=tokenizer.decode(new,skip_special_tokens=True)
                code=extract_code(readable,row['problem'])
                check=check_solution(code,row['problem'],run_dir/'sandbox'/(str(i)+'_'+uuid.uuid4().hex))
                result=dict(id=row['id'],raw_generation=raw,code=code,correct=bool(check['passed']),error=check['error'],generated_tokens=len(new),hit_token_limit=len(new)>=CFG['max_new_tokens'])
            result['seconds']=time.monotonic()-start
            stream.write(json.dumps(result,ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            done[str(row['id'])]=result
            if (i+1)%10==0 or i+1==len(rows): status('evaluating',run=run_dir.name,dataset=key,completed=len(done),total=len(rows))
    assert len(done)==len(rows)
    correct=sum(bool(r['correct']) for r in done.values())
    return dict(correct=correct,total=len(rows),accuracy=correct/len(rows))

def run_all_benchmarks(model,tokenizer,prompts,label):
    run_dir=OUT/label; run_dir.mkdir(parents=True,exist_ok=True)
    results={key:evaluate_dataset(model,tokenizer,key,rows,run_dir) for key,rows in prompts.items()}
    dump(run_dir/'metrics.json',results)
    return results



def summarize_results():
    records=[]
    for seed in CFG['fault_seeds']:
        directory=OUT/f'seed_{seed:03d}'
        path=directory/'metrics.json'
        if not path.exists(): continue
        metrics=json.loads(path.read_text())
        fault=json.loads((directory/'faults'/'fault_manifest.json').read_text())
        record=dict(seed=seed,actual_ber=fault['actual_ber'],flipped_bits=fault['actual_flips'],corrupted_weights=fault['corrupted_weights'])
        record.update({key:metrics[key]['accuracy']*100 for key in ['mathqa','mmlu','humaneval']})
        records.append(record)
    if not records: return
    frame=pd.DataFrame(records); frame.to_csv(OUT/'runs.csv',index=False)
    baseline={}
    for label in ['bf16','clean_w8a16']:
        path=OUT/label/'metrics.json'
        if path.exists(): baseline[label]=json.loads(path.read_text())
    summary=[]; cumulative=[]
    for key in ['mathqa','mmlu','humaneval']:
        a=frame[key].to_numpy(); n=len(a); mean=float(a.mean())
        std=float(a.std(ddof=1)) if n>1 else None
        se=std/math.sqrt(n) if std is not None else None
        half=float(student_t.ppf(.975,n-1)*se) if n>1 else None
        target=CFG['ci_target_pp']
        estimated=None
        if std is not None and std>0:
            estimated=max(10,n)
            while student_t.ppf(.975,estimated-1)*std/math.sqrt(estimated)>target and estimated<10000: estimated+=1
        clean=baseline.get('clean_w8a16',{}).get(key,{}).get('accuracy')
        bf=baseline.get('bf16',{}).get(key,{}).get('accuracy')
        summary.append(dict(dataset=key,n=n,bf16_pct=None if bf is None else bf*100,clean_w8a16_pct=None if clean is None else clean*100,fault_mean_pct=mean,std_pp=std,se_pp=se,ci95_low_pct=None if half is None else mean-half,ci95_high_pct=None if half is None else mean+half,ci95_halfwidth_pp=half,fault_minus_clean_pp=None if clean is None else mean-clean*100,precision_target_pp=target,pilot_precision_met=bool(n>=10 and std is not None and std>0 and half<=target),estimated_total_runs=estimated))
        for k in range(1,n+1):
            s=float(a[:k].std(ddof=1)) if k>1 else None
            h=float(student_t.ppf(.975,k-1)*s/math.sqrt(k)) if k>1 else None
            cumulative.append(dict(dataset=key,n=k,mean_pct=float(a[:k].mean()),std_pp=s,ci_halfwidth_pp=h))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(cumulative).to_csv(OUT/'cumulative.csv',index=False)
    dump(OUT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key in zip(axes,['mathqa','mmlu','humaneval']):
        points=[r for r in cumulative if r['dataset']==key]
        x=np.array([r['n'] for r in points]); y=np.array([r['mean_pct'] for r in points])
        h=np.array([r['ci_halfwidth_pp'] or 0 for r in points])
        ax.plot(x,y,'o-'); ax.fill_between(x,y-h,y+h,alpha=.2)
        ax.set(title=key,xlabel='Independent fault runs',ylabel='Cumulative accuracy (%)')
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'cumulative_mean.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,r in zip(axes,summary):
        values=[r['bf16_pct'],r['clean_w8a16_pct'],r['fault_mean_pct']]
        ax.bar(['BF16','W8A16 clean','Fault mean'],[np.nan if v is None else v for v in values],yerr=[0,0,r['std_pp'] or 0],capsize=4)
        ax.set(title=r['dataset'],ylabel='Accuracy (%)',ylim=(0,100)); ax.tick_params(axis='x',rotation=20)
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)



def main():
    status('preflight')
    torch.set_num_threads(4)
    torch.manual_seed(1234); np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    quantization_self_test(); fault_self_test(); sandbox_self_test()
    data,by_subject=load_data()
    if '--self-test' in sys.argv:
        status('self_tests_passed'); return
    assert torch.cuda.is_available(),'CUDA 不可用'
    device=CFG['device_index']; torch.cuda.set_device(device)
    assert torch.cuda.is_bf16_supported(),'设备不支持BF16'
    free,total=torch.cuda.mem_get_info(device)
    assert free>32*1024**3 if CFG['cache_dequant'] else free>22*1024**3, '空闲GPU显存不足，请等待或关闭cache_dequant'
    assert shutil.disk_usage(ROOT).free> (25+9*len(CFG['fault_seeds']))*1024**3, '磁盘不足：模型和每轮故障mask需要空间'
    from huggingface_hub import HfApi
    revision_path=OUT/'model_revision.json'
    if CFG['local_model']:
        model_source=str(safe(CFG['local_model'])); revision=None
    else:
        model_source=CFG['model_id']
        if revision_path.exists(): revision=json.loads(revision_path.read_text())['revision']
        else:
            revision=HfApi().model_info(model_source,revision=CFG['model_revision']).sha
            dump(revision_path,dict(model_id=model_source,revision=revision))
    status('loading_model',model=model_source,revision=revision)
    kwargs=dict(cache_dir=str(safe('.runtime/hf/hub')),revision=revision,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(model_source,**kwargs)
    tokenizer.pad_token=tokenizer.eos_token
    prompts=build_prompts(tokenizer,data,by_subject)
    model=AutoModelForCausalLM.from_pretrained(model_source,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':device},**kwargs).eval()
    assert model.config.model_type=='qwen3'
    assert model.config.max_position_embeddings>=CFG['context_limit']
    versions={name:__import__(name).__version__ for name in ['torch','transformers','numpy','datasets','scipy']}
    manifest=dict(config=CFG,revision=revision,code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions=versions,prompt_hashes={k:digest(v) for k,v in prompts.items()},gpu=torch.cuda.get_device_name(device))
    # 允许追加种子；其他运行协议必须一致。
    stable=dict(manifest); stable['config']={k:v for k,v in CFG.items() if k not in ['fault_seeds','ci_target_pp']}
    manifest_path=OUT/'experiment_manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        assert existing==stable,'实验配置/代码/数据发生变化，请改 result_name 新建实验'
    else: dump(manifest_path,stable)
    dump(OUT/'requested_runs.json',CFG)
    # 单条选择题端到端冒烟测试，不写入正式指标。
    scores=choice_scores(model,tokenizer,prompts['mathqa'][0])
    assert scores is not None and all(math.isfinite(s) for s in scores)
    if CFG['bf16_baseline']:
        status('bf16_baseline'); run_all_benchmarks(model,tokenizer,prompts,'bf16')
    status('quantizing')
    layers=replace_linear_with_quantlinear(model); gc.collect(); torch.cuda.empty_cache()
    restore_clean_qweights(layers)
    assert all(m.qweight.dtype==torch.int8 for _,m in layers)
    status('clean_w8a16'); run_all_benchmarks(model,tokenizer,prompts,'clean_w8a16')
    for seed in CFG['fault_seeds']:
        folder=OUT/f'seed_{seed:03d}'
        if (folder/'metrics.json').exists(): continue
        status('injecting_fault',seed=seed)
        info=inject_random_bit_errors(layers,seed,folder/'faults')
        print('Fault verified:',{k:v for k,v in info.items() if k!='layers'},flush=True)
        try: run_all_benchmarks(model,tokenizer,prompts,folder.name)
        finally: restore_clean_qweights(layers)
        summarize_results()
    summarize_results()
    status('complete',fault_runs=len(CFG['fault_seeds']))


# ===== v2: weight-error-rate 0.003 + FP16 + MathQA rationale =====
original_build_prompts = build_prompts
original_evaluate_dataset = evaluate_dataset

def build_prompts(tokenizer,data,by_subject):
    prompts=original_build_prompts(tokenizer,data,by_subject)
    rows=[]
    for i,r in enumerate(data['mathqa_fixed_1000']):
        messages=[dict(role='system',content='Solve the multiple-choice math problem step by step. End your response with exactly: Final answer: X, where X is A, B, C, D, or E.')]
        for s in data['mathqa_fixed_5shot']:
            messages += [dict(role='user',content=math_question(s)),dict(role='assistant',content=s['Rationale'].strip()+'\nFinal answer: '+s['correct'].strip().upper())]
        messages.append(dict(role='user',content=math_question(r)))
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert len(tokenizer.encode(prompt,add_special_tokens=False))+CFG['mathqa_max_new_tokens']<=CFG['context_limit'],'MathQA prompt太长，禁止静默截断'
        rows.append(dict(id=i,prompt=prompt,gold=r['correct'].strip().upper(),labels=list('ABCDE'),subject=None))
    prompts['mathqa']=rows
    dump(OUT/'prompt_manifest.json',{k:dict(count=len(v),sha256=digest(v)) for k,v in prompts.items()})
    return prompts

@torch.inference_mode()
def evaluate_dataset(model,tokenizer,key,rows,run_dir):
    if key!='mathqa': return original_evaluate_dataset(model,tokenizer,key,rows,run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    path=run_dir/'mathqa.jsonl'
    prior=read_jsonl(path); done={str(r['id']):r for r in prior}
    assert len(done)==len(prior) and set(done)<={str(r['id']) for r in rows}
    pending=[r for r in rows if str(r['id']) not in done]
    tokenizer.padding_side='left'
    with path.open('a',buffering=1) as stream:
        for start in range(0,len(pending),CFG['mathqa_batch_size']):
            batch=pending[start:start+CFG['mathqa_batch_size']]
            encoded=tokenizer([r['prompt'] for r in batch],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
            generated=model.generate(**encoded,do_sample=False,num_beams=1,max_new_tokens=CFG['mathqa_max_new_tokens'],pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            for row,seq in zip(batch,generated):
                ids=seq[encoded.input_ids.shape[1]:].tolist()
                if tokenizer.eos_token_id in ids: ids=ids[:ids.index(tokenizer.eos_token_id)]
                text=tokenizer.decode(ids,skip_special_tokens=True)
                matches=re.findall(r'(?im)^\s*Final\s+answer\s*:\s*\**\s*([A-E])\b',text)
                prediction=matches[-1].upper() if matches else None
                record=dict(id=row['id'],gold=row['gold'],prediction=prediction,correct=prediction==row['gold'],generation=text,parse_failed=prediction is None,generated_tokens=len(ids),hit_token_limit=len(ids)>=CFG['mathqa_max_new_tokens'])
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                done[str(row['id'])]=record
            status('evaluating',run=run_dir.name,dataset=key,completed=len(done),total=len(rows),correct=sum(r['correct'] for r in done.values()))
    assert len(done)==len(rows)
    n=len(rows); correct=sum(r['correct'] for r in done.values())
    return dict(correct=correct,total=n,accuracy=correct/n,parse_failures=sum(r['parse_failed'] for r in done.values()),truncated=sum(r['hit_token_limit'] for r in done.values()))

@torch.no_grad()
def inject_weight_errors(layers,seed,folder):
    restore_clean_qweights(layers)
    folder=safe(folder); folder.mkdir(parents=True,exist_ok=True)
    sizes=np.array([m.qweight.numel() for _,m in layers],dtype=np.int64)
    ends=np.cumsum(sizes); total=int(ends[-1]); k=round(total*CFG['weight_error_rate'])
    rng=np.random.default_rng(seed)
    selected=np.sort(rng.choice(total,k,replace=False,shuffle=False))
    bits=rng.integers(0,8,size=k,dtype=np.uint8)
    assert len(selected)==k and (k<2 or np.all(selected[1:]>selected[:-1]))
    np.save(folder/'global_weight_indices.npy',selected)
    np.save(folder/'bit_positions.npy',bits)
    actual=0; changed_count=0; bit_counts=np.zeros(8,dtype=np.int64); metadata=[]
    popcount=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    base=0
    for name,m in layers:
        m._cache=None
        clean=m.clean.numpy().view(np.uint8).reshape(-1); flat=m.qweight.reshape(-1)
        layer_changed=0
        for offset in range(0,len(clean),1048576):
            end=min(offset+1048576,len(clean))
            left=int(np.searchsorted(selected,base+offset)); right=int(np.searchsorted(selected,base+end))
            positions=selected[left:right]-base-offset; block_bits=bits[left:right]
            values=clean[offset:end].copy()
            values[positions]^=(1<<block_bits).astype(np.uint8)
            flat[offset:end].copy_(torch.from_numpy(values.view(np.int8)).to(flat.device))
            diff=np.bitwise_xor(flat[offset:end].cpu().numpy().view(np.uint8),clean[offset:end])
            flips=int(popcount[diff].sum()); changed=int(np.count_nonzero(diff))
            assert flips==changed==right-left,'每个选中权重必须且只能翻转1位'
            actual+=flips; changed_count+=changed; layer_changed+=changed
            bit_counts+=np.bincount(block_bits,minlength=8)
        assert torch.equal(m.scale.cpu(),m.scale_clean),name
        m.refresh()
        metadata.append(dict(layer=name,shape=list(m.qweight.shape),global_weight_start=base,num_weights=len(clean),corrupted_weights=layer_changed))
        base+=len(clean)
    assert actual==changed_count==k and int(bit_counts.sum())==k
    info=dict(seed=seed,target_weight_error_rate=CFG['weight_error_rate'],actual_weight_error_rate=changed_count/total,total_int8_weights=total,total_bits=8*total,expected_flips=k,actual_flips=actual,corrupted_weights=changed_count,actual_ber=actual/(8*total),bit_counts=bit_counts.tolist(),layers=metadata)
    dump(folder/'fault_manifest.json',info)
    print('VERIFIED weight injection:',{a:b for a,b in info.items() if a!='layers'},flush=True)
    return info

def v2_self_test():
    quantization_self_test(); sandbox_self_test()
    layers=[('toy',QuantLinear(nn.Linear(259,17,dtype=torch.bfloat16)))]
    clean=layers[0][1].qweight.clone()
    location=OUT/'self_tests'/uuid.uuid4().hex
    info=inject_weight_errors(layers,101,location/'a')
    assert info['actual_flips']==round(clean.numel()*.003)
    first=layers[0][1].qweight.clone()
    inject_weight_errors(layers,101,location/'b')
    assert torch.equal(first,layers[0][1].qweight)
    restore_clean_qweights(layers)
    assert torch.equal(clean,layers[0][1].qweight)
    dump(OUT/'self_test.json',dict(passed=True,definition='round(N * 0.003) distinct weights, exactly one random bit each'))
    print('PASS v2: 不同权重抽样、每权重1位、数量精确、复现和恢复',flush=True)

def v2_summary():
    summarize_results()
    baseline={}
    for label in ['fp16','bf16','clean_w8a16','fp16_original_mathqa']:
        p=OUT/label/'metrics.json'
        if p.exists(): baseline[label]=json.loads(p.read_text())
    dump(OUT/'baselines.json',baseline)
    summary=OUT/'summary.csv'
    if summary.exists() and 'fp16' in baseline:
        df=pd.read_csv(summary)
        df['fp16_pct']=[baseline['fp16'][key]['accuracy']*100 for key in df['dataset']]
        df.to_csv(summary,index=False)

def run_v2():
    status('preflight_v2')
    torch.set_num_threads(4); torch.manual_seed(1234); np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.benchmark=False
    v2_self_test()
    data,by_subject=load_data()
    assert torch.cuda.is_available()
    device=CFG['device_index']; torch.cuda.set_device(device)
    assert torch.cuda.mem_get_info(device)[0]>32*1024**3,'GPU空间不足，请等待后续重跑'
    old=ROOT/'results_int8/qwen3_8b_w8a16_g128_ber003_v1'
    revision=json.loads((old/'model_revision.json').read_text())['revision']
    kwargs=dict(cache_dir=str(safe('.runtime/hf/hub')),revision=revision,trust_remote_code=False,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(CFG['model_id'],**kwargs)
    tokenizer.pad_token=tokenizer.eos_token
    original=original_build_prompts(tokenizer,data,by_subject)
    prompts=build_prompts(tokenizer,data,by_subject)
    manifest=dict(config=CFG,revision=revision,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prompt_hashes={k:digest(v) for k,v in prompts.items()})
    p=OUT/'v2_manifest.json'
    if p.exists(): assert json.loads(p.read_text())==manifest,'v2配置变化，必须另建结果目录'
    else: dump(p,manifest)
    def load(dtype):
        status('loading_model',dtype=str(dtype))
        return AutoModelForCausalLM.from_pretrained(CFG['model_id'],torch_dtype=dtype,attn_implementation='sdpa',device_map={'':device},**kwargs).eval()
    # 首先完成用户要求的FP16基线；旧MathQA协议另外保存，便于区分精度和提示词影响。
    if not (OUT/'fp16/metrics.json').exists():
        model=load(torch.float16)
        old_dir=OUT/'fp16_original_mathqa'; old_dir.mkdir(parents=True,exist_ok=True)
        metric=original_evaluate_dataset(model,tokenizer,'mathqa',original['mathqa'],old_dir)
        dump(old_dir/'metrics.json',dict(mathqa=metric))
        run_all_benchmarks(model,tokenizer,prompts,'fp16'); v2_summary()
        del model; gc.collect(); torch.cuda.empty_cache()
    model=load(torch.bfloat16)
    run_all_benchmarks(model,tokenizer,prompts,'bf16'); v2_summary()
    status('quantizing')
    layers=replace_linear_with_quantlinear(model); gc.collect(); torch.cuda.empty_cache()
    restore_clean_qweights(layers)
    if not CFG.get('skip_clean_w8a16',False):
        run_all_benchmarks(model,tokenizer,prompts,'clean_w8a16')
    else:
        status('skipping_clean_w8a16',reason='existing_compatible_baseline')
    v2_summary()
    for seed in CFG['fault_seeds']:
        directory=OUT/f'seed_{seed:03d}'
        if (directory/'metrics.json').exists(): continue
        status('injecting_weight_errors',seed=seed)
        inject_weight_errors(layers,seed,directory/'faults')
        try: run_all_benchmarks(model,tokenizer,prompts,directory.name)
        finally: restore_clean_qweights(layers)
        v2_summary()
    v2_summary(); status('complete',fault_runs=len(CFG['fault_seeds']))


class ResearchLinear(nn.Module):
    def __init__(self, linear, bits):
        super().__init__()
        self.in_features=linear.in_features; self.out_features=linear.out_features
        self.bits=bits; self.group_size=128
        self.register_buffer('bias',None if linear.bias is None else linear.bias.detach().cpu().to(torch.bfloat16).clone())
        self.register_buffer('_cache',None,persistent=False)
        self.target_device=torch.device('cpu')
        qs=[]; scales=[]; bound=2**(bits-1)-1
        for start in range(0,self.out_features,128):
            w=linear.weight[start:start+128].detach().cpu().float()
            assert torch.isfinite(w).all()
            p=F.pad(w,(0,(-self.in_features)%128)).reshape(w.shape[0],-1,128)
            mx=p.abs().amax(-1); sc=torch.where(mx>0,mx/bound,torch.ones_like(mx))
            q=(p/sc.unsqueeze(-1)).round().clamp(-bound,bound).to(torch.int8)
            qs.append(q.reshape(w.shape[0],-1)[:,:self.in_features].contiguous()); scales.append(sc)
        raw=torch.cat(qs); self.scale=torch.cat(scales); self.scale_clean=self.scale.clone()
        if bits==8: self.qweight=raw
        else:
            u=(raw.reshape(-1).to(torch.int16)&15).to(torch.uint8)
            if u.numel()%2: u=F.pad(u,(0,1))
            self.qweight=(u[0::2]|(u[1::2]<<4)).contiguous()
        self.clean=self.qweight.clone()
    def integer_rows(self,start,end):
        if self.bits==8: return self.qweight[start:end]
        lo=start*self.in_features; hi=end*self.in_features
        packed=self.qweight[lo//2:(hi+1)//2]
        u=torch.stack((packed&15,packed>>4),dim=1).flatten()
        u=u[lo%2:lo%2+hi-lo].to(torch.int8)
        return torch.where(u>=8,u-16,u).reshape(end-start,self.in_features)
    @torch.no_grad()
    def dequantize_weight(self):
        out=torch.empty((self.out_features,self.in_features),dtype=torch.bfloat16,device=self.target_device)
        for start in range(0,self.out_features,128):
            end=min(start+128,self.out_features)
            q=self.integer_rows(start,end)
            p=F.pad(q,(0,(-self.in_features)%128)).reshape(end-start,-1,128)
            w=(p.float()*self.scale[start:end,:,None]).reshape(end-start,-1)[:,:self.in_features].to(torch.bfloat16)
            out[start:end].copy_(w)
        return out
    @torch.no_grad()
    def refresh(self):
        self._cache=None
        self._cache=self.dequantize_weight()
    def forward(self,x):
        assert self._cache is not None
        return F.linear(x.to(torch.bfloat16),self._cache,self.bias)

def quantize_cpu(model,bits):
    names=[n for n,m in model.named_modules() if isinstance(m,nn.Linear)]
    layers=[]; report=[]
    for name in names:
        linear=model.get_submodule(name)
        layer=ResearchLinear(linear,bits)
        ref=linear.weight[:8].detach().float()
        q=layer.integer_rows(0,min(8,layer.out_features))
        p=F.pad(q,(0,(-layer.in_features)%128)).reshape(q.shape[0],-1,128)
        recon=(p.float()*layer.scale[:q.shape[0],:,None]).reshape(q.shape[0],-1)[:,:layer.in_features]
        report.append(dict(name=name,shape=[layer.out_features,layer.in_features],bits=bits,integer_storage_bytes=layer.qweight.numel(),mse=float((ref-recon).square().mean())))
        parent,_,leaf=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,leaf,layer)
        layers.append((name,layer)); del linear,ref,recon,q,p
    assert not any(isinstance(m,nn.Linear) for m in model.modules())
    dump(OUT/f'quantization_int{bits}.json',report)
    return layers

def make_fault_masks(layers, seed, folder, ber):
    folder=safe(folder); folder.mkdir(parents=True,exist_ok=True)
    sizes=np.array([m.qweight.numel() for _,m in layers],dtype=np.int64)
    boundaries=np.cumsum(sizes)*8
    total=int(boundaries[-1]); target=int(round(total*ber))
    assert 0<=ber<=1 and total>0
    rng=np.random.default_rng(seed)
    masks=[]; selected=0; block=262144
    for j,((name,m),size) in enumerate(zip(layers,sizes)):
        path=folder/f'{j:04d}.mask'
        mask=np.zeros(int(size),dtype=np.uint8)
        for start in range(0,int(size),block):
            n=min(block,int(size)-start); count=int(rng.binomial(n*8,ber))
            ids=rng.choice(n*8,count,replace=False)
            values=np.zeros(n,dtype=np.uint8)
            np.bitwise_or.at(values,ids//8,(1<<(ids%8)).astype(np.uint8))
            mask[start:start+n]=values; selected+=count
        masks.append(mask)
    # 均匀地从已选位中移除，或从未选位中补充；同一批先去重。
    delta=target-selected
    while delta:
        want_on=delta<0
        acceptance=ber if want_on else 1-ber
        count=min(2000000,max(1024,int(abs(delta)/max(acceptance,1e-9)*1.3)))
        candidates=np.unique(rng.integers(0,total,size=count,dtype=np.int64))
        rng.shuffle(candidates)
        for g in candidates:
            layer=int(np.searchsorted(boundaries,g,side='right'))
            local=int(g-(int(boundaries[layer-1]) if layer else 0))
            index,bit=divmod(local,8); flag=np.uint8(1<<bit)
            is_on=bool(masks[layer][index]&flag)
            if is_on==want_on:
                masks[layer][index]^=flag
                delta += 1 if want_on else -1
                if delta==0: break
    for j,mask in enumerate(masks): np.savez_compressed(folder/f'{j:04d}.mask.npz',mask=mask)
    return masks,total,target





def evaluate_dataset(model,tokenizer,key,rows,run_dir):
    if key!='mathqa': return original_evaluate_dataset(model,tokenizer,key,rows,run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    path=run_dir/'mathqa.jsonl'
    prior=read_jsonl(path); done={str(r['id']):r for r in prior}
    assert len(done)==len(prior) and set(done)<={str(r['id']) for r in rows}
    pending=[r for r in rows if str(r['id']) not in done]
    tokenizer.padding_side='left'
    with path.open('a',buffering=1) as stream:
        for start in range(0,len(pending),CFG['mathqa_batch_size']):
            batch=pending[start:start+CFG['mathqa_batch_size']]
            encoded=tokenizer([r['prompt'] for r in batch],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
            generated=model.generate(**encoded,do_sample=False,num_beams=1,max_new_tokens=CFG['mathqa_max_new_tokens'],pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            for row,seq in zip(batch,generated):
                ids=seq[encoded.input_ids.shape[1]:].tolist()
                if tokenizer.eos_token_id in ids: ids=ids[:ids.index(tokenizer.eos_token_id)]
                text=tokenizer.decode(ids,skip_special_tokens=True)
                normalized_text=text.replace("*","")
                matches=re.findall(r'(?im)^\s*(?:#{1,6}\s*)?Final\s+answer\s*:\s*\**\s*([A-E])\b',normalized_text)
                prediction=matches[-1].upper() if matches else None
                record=dict(id=row['id'],gold=row['gold'],prediction=prediction,correct=prediction==row['gold'],generation=text,parse_failed=prediction is None,generated_tokens=len(ids),hit_token_limit=len(ids)>=CFG['mathqa_max_new_tokens'])
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                done[str(row['id'])]=record
            status('evaluating',run=run_dir.name,dataset=key,completed=len(done),total=len(rows),correct=sum(r['correct'] for r in done.values()))
    assert len(done)==len(rows)
    n=len(rows); correct=sum(r['correct'] for r in done.values())
    return dict(correct=correct,total=n,accuracy=correct/n,parse_failures=sum(r['parse_failed'] for r in done.values()),truncated=sum(r['hit_token_limit'] for r in done.values()))


# Existing run loop calls this name; BER arg is intentionally not used.




# Rejecting proposals to already-faulted weights gives a uniform draw over the
# remaining weights. Those rejected proposals are NOT sampling attempts.
# Within a batch retain all zero attempts BEFORE each weight's first success.
def consume_proposals(clean,faulted,ids,bits,remaining):
    pos=np.arange(len(ids)); eligible=~faulted[ids]
    hit=(clean[ids]&np.left_shift(np.uint8(1),bits))!=0
    potential=np.flatnonzero(eligible&hit)
    cutoff=len(ids)
    valid=eligible.copy()
    if len(potential):
        keys,first=np.unique(ids[potential],return_index=True)
        firstpos=potential[first]
        lookup=np.searchsorted(keys,ids); clipped=np.minimum(lookup,len(keys)-1)
        matched=(lookup<len(keys))&(keys[clipped]==ids)
        valid &= (~matched)|(pos<=firstpos[clipped])
        if len(firstpos)>=remaining:
            cutoff=int(np.partition(firstpos,remaining-1)[remaining-1])+1
            valid &= pos<cutoff
    wins=valid&hit
    return valid,wins,cutoff

@torch.no_grad()
def inject_effective_global(layers,seed,folder,target_ber=None):
    target_ber=CFG['target_BER'] if target_ber is None else target_ber
    assert 0<=target_ber<=.125
    folder=safe(folder); folder.mkdir(parents=True,exist_ok=True)
    sizes=np.array([m.qweight.numel() for _,m in layers],dtype=np.int64)
    ends=np.cumsum(sizes); N=int(ends[-1]); total_bits=8*N; target=round(total_bits*target_ber)
    for name,m in layers:
        assert m.bits==8 and m.qweight.dtype==torch.int8 and m.qweight.device.type=='cpu'
        m.qweight.copy_(m.clean); assert torch.equal(m.qweight,m.clean)
        assert torch.equal(m.scale,m.scale_clean)
    clean=np.concatenate([m.clean.numpy().view(np.uint8).reshape(-1) for _,m in layers])
    assert target<=int(np.count_nonzero(clean)), 'Not enough nonzero weights for one effective fault per weight'
    faulted=np.zeros(N,dtype=np.bool_)
    chosen_ids=np.empty(target,dtype=np.int64); chosen_bits=np.empty(target,dtype=np.uint8)
    attempts=np.zeros(8,dtype=np.int64); clears=attempts.copy()
    layer_attempts=np.zeros(len(layers),dtype=np.int64)
    rng=np.random.default_rng(int(seed)); count=0; proposals=0; batches=0
    while count<target:
        size=min(1048576,max(4096,2*(target-count)))
        ids=rng.integers(0,N,size=size,dtype=np.int64)
        bits=rng.integers(0,8,size=size,dtype=np.uint8)
        valid,wins,cutoff=consume_proposals(clean,faulted,ids,bits,target-count)
        ii=ids[wins]; bb=bits[wins]; k=len(ii)
        chosen_ids[count:count+k]=ii; chosen_bits[count:count+k]=bb
        faulted[ii]=True; count+=k; proposals+=cutoff; batches+=1
        attempts+=np.bincount(bits[valid],minlength=8)
        clears+=np.bincount(bb,minlength=8)
        layer_attempts+=np.bincount(np.searchsorted(ends,ids[valid],side='right'),minlength=len(layers))
        if batches%32==0: print('INJECTION_PROGRESS',dict(seed=seed,effective=count,target=target,attempts=int(attempts.sum())),flush=True)
    assert int(faulted.sum())==target==int(clears.sum())
    del faulted,clean
    order=np.argsort(chosen_ids); chosen_ids=chosen_ids[order]; chosen_bits=chosen_bits[order]; del order
    assert target<2 or np.all(chosen_ids[1:]>chosen_ids[:-1])
    np.save(folder/'faulted_global_weight_indices.npy',chosen_ids)
    np.save(folder/'faulted_bit_positions.npy',chosen_bits)
    pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    rows=[]; offset=0; actual_total=0
    for layer_index,((name,m),n) in enumerate(zip(layers,sizes)):
        n=int(n); a=np.searchsorted(chosen_ids,offset); b=np.searchsorted(chosen_ids,offset+n)
        ids=chosen_ids[a:b]-offset; bits=chosen_bits[a:b]; masks=np.left_shift(np.uint8(1),bits)
        clean=m.clean.numpy().view(np.uint8).reshape(-1); q=m.qweight.numpy().view(np.uint8).reshape(-1)
        assert np.all((clean[ids]&masks)!=0)
        q[ids]=np.bitwise_and(clean[ids],np.bitwise_not(masks))
        actual=0; unique=0
        for start in range(0,n,1048576):
            end=min(n,start+1048576); lo=np.searchsorted(ids,start); hi=np.searchsorted(ids,end)
            expected=clean[start:end].copy(); expected[ids[lo:hi]-start]&=np.bitwise_not(masks[lo:hi])
            assert np.array_equal(q[start:end],expected),name
            assert not np.any(q[start:end]&np.bitwise_not(clean[start:end])), '0->1 forbidden'
            removed=clean[start:end]&np.bitwise_not(q[start:end])
            assert np.all(pop[removed]<=1),'More than one effective fault per weight'
            actual+=int(pop[removed].sum()); unique+=int(np.count_nonzero(removed))
        assert actual==unique==len(ids)
        actual_total+=actual
        row=dict(tensor_name=name,layer_index=layer_index,num_weights=n,global_weight_offset=offset,
                 sampling_attempts=int(layer_attempts[layer_index]),effective_flips=actual,effective_BER_in_tensor=actual/(8*n))
        rows.append(row); offset+=n
        assert torch.equal(m.scale,m.scale_clean); m.refresh()
    assert actual_total==target
    metadata=dict(experiment_name='HBM_effective_BER_0.003_1to0',seed=int(seed),
        target_BER=target_ber,BER_definition='actual_changed_bits / total_int8_bits',fault_direction='1_to_0',
        resample_if_selected_bit_is_zero=False,max_effective_faults_per_weight=None,sampling_scope='global_model',
        total_int8_weights=N,total_bits=total_bits,target_error_bits=target,
        sampling_attempts=int(attempts.sum()),effective_1to0_flips=actual_total,effective_0to1_flips=0,
        effective_BER=actual_total/total_bits,unique_faulted_weights=actual_total,
        zero_bit_attempts=int(attempts.sum())-actual_total,
        rejected_already_faulted_proposals=proposals-int(attempts.sum()),
        bit_index_definition='bit0 is LSB; bit7 is INT8 two-complement MSB/sign bit',
        msb_attempts=int(attempts[7]),msb_effective_flips=int(clears[7]),
        clean_restored_and_verified=True,layers=rows)
    for bit in range(8):
        metadata[f'bit{bit}_attempts']=int(attempts[bit])
        metadata[f'bit{bit}_effective_flips']=int(clears[bit])
    assert sum(metadata[f'bit{i}_attempts'] for i in range(8))==metadata['sampling_attempts']
    assert sum(metadata[f'bit{i}_effective_flips'] for i in range(8))==target
    assert abs(metadata['effective_BER']-target_ber)<=.5/total_bits+1e-16
    dump(folder/'fault_manifest.json',metadata)
    pd.DataFrame(rows).to_csv(folder/'tensor_stats.csv',index=False)
    pd.DataFrame([dict(bit=i,attempts=int(attempts[i]),effective_flips=int(clears[i])) for i in range(8)]).to_csv(folder/'bit_stats.csv',index=False)
    print('FAULT_VERIFIED',json.dumps({k:v for k,v in metadata.items() if k!='layers'}),flush=True)
    return metadata

def test_effective_global():
    # Compare batched processing to literal sequential rules, including revisits.
    clean=np.array([0,128,255,1,15],dtype=np.uint8)
    rng=np.random.default_rng(908)
    for trial in range(80):
        ids=rng.integers(0,5,100); bits=rng.integers(0,8,100,dtype=np.uint8)
        initial=np.array([False,False,trial%2==0,False,False]); remaining=2
        valid,wins,cutoff=consume_proposals(clean,initial,ids,bits,remaining)
        seen=initial.copy(); vv=np.zeros(100,dtype=bool); ww=vv.copy(); count=0; end=100
        for j,(i,b) in enumerate(zip(ids,bits)):
            if seen[i]: continue
            vv[j]=True
            if int(clean[i])&(1<<int(b)):
                ww[j]=True; seen[i]=True; count+=1
                if count==remaining: end=j+1; break
        assert np.array_equal(valid,vv) and np.array_equal(wins,ww) and cutoff==end
    # Explicitly prove a zero trial doesn't remove that weight from eligibility.
    v,w,c=consume_proposals(np.array([128],dtype=np.uint8),np.zeros(1,dtype=bool),np.array([0,0,0]),np.array([0,7,7],dtype=np.uint8),1)
    assert v.tolist()==[True,True,False] and w.tolist()==[False,True,False]
    class Toy:
        bits=8
        def __init__(self,values):
            self.clean=torch.tensor(values,dtype=torch.int8).reshape(1,-1); self.qweight=self.clean.clone()
            self.scale=torch.ones(1); self.scale_clean=self.scale.clone()
        def refresh(self): pass
    def toys(): return [('a',Toy(list(range(-128,128))*2)),('b',Toy([-1]*512))]
    xs=toys(); m=inject_effective_global(xs,77,OUT/'tests/a')
    assert m['effective_1to0_flips']==round(1024*8*.003)
    saved=[x.qweight.clone() for _,x in xs]
    inject_effective_global(xs,77,OUT/'tests/replay')
    assert all(torch.equal(a,x.qweight) for a,(_,x) in zip(saved,xs))
    inject_effective_global(xs,78,OUT/'tests/next'); fresh=toys()
    inject_effective_global(fresh,78,OUT/'tests/fresh')
    assert all(torch.equal(x.qweight,y.qweight) for (_,x),(_,y) in zip(xs,fresh))
    try: inject_effective_global([('zeros',Toy([0]*1000))],0,OUT/'tests/impossible')
    except AssertionError: pass
    else: raise AssertionError('Impossible target must fail')
    sandbox_self_test()
    dump(OUT/'self_test_v6.json',dict(passed=True,tests=['batch equals sequential sampler','zero weight remains eligible','one effective fault per weight','global exact integer BER','no 0to1','restore and seed replay','impossible target','HumanEval sandbox']))

def summary_v6():
    records=[]
    for seed in CFG['fault_seeds']:
        p=OUT/f'seed_{seed:03d}/metrics.json'
        if p.exists(): records.append((seed,json.loads(p.read_text())))
    report={'completed_seeds':[s for s,_ in records],'std_definition':'sample standard deviation (ddof=1), percentage points','datasets':{}}
    rows=[]
    for key in ('mathqa','mmlu','humaneval'):
        vals=[m[key]['accuracy']*100 for _,m in records]
        stat=lambda v:dict(n=len(v),mean_pct=float(np.mean(v)) if v else None,std_pp=float(np.std(v,ddof=1)) if len(v)>1 else None)
        groups=[]
        for first in range(0,len(records)-4,5):
            g=records[first:first+5]; groups.append(dict(seeds=[s for s,_ in g],**stat([m[key]['accuracy']*100 for _,m in g])))
        clean=json.loads((OUT/'clean_w8a16/metrics.json').read_text())[key]['accuracy']*100 if (OUT/'clean_w8a16/metrics.json').exists() else None
        report['datasets'][key]=dict(clean_baseline_pct=clean,mean_accuracy_drop_pp=clean-float(np.mean(vals)) if vals and clean is not None else None,all_completed=stat(vals),five_run_groups=groups,ten_run_average=stat(vals[:10]) if len(vals)>=10 else None)
        for seed,m in records: rows.append(dict(seed=seed,dataset=key,**m[key]))
    dump(OUT/'summary.json',report); pd.DataFrame(rows).to_csv(OUT/'runs.csv',index=False)
    print('ACCURACY_SUMMARY',json.dumps(report),flush=True)
    return report

def uniform_subset_reference(n, k, seed):
    assert 0 <= k <= n
    if k == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected = np.unique(rng.integers(0, n, size=k, dtype=np.int64))
    while selected.size < k:
        selected = np.union1d(selected, rng.integers(0, n, size=k-selected.size, dtype=np.int64))
    return selected

def reference_self_test():
    popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    raw = np.random.default_rng(20260918).integers(0, 256, size=8192, dtype=np.uint8)
    total_ones = int(popcount[raw].sum(dtype=np.int64))
    k = 257
    chosen = uniform_subset_reference(total_ones, k, 17)
    one_positions = np.flatnonzero(np.unpackbits(raw, bitorder='little'))
    before = raw.copy()
    selected_positions = one_positions[chosen]
    mask = np.zeros_like(raw, dtype=np.uint8)
    np.bitwise_or.at(mask, selected_positions // 8, (1 << (selected_positions % 8)).astype(np.uint8))
    after = before & np.bitwise_not(mask)
    assert not np.any(np.bitwise_and(np.bitwise_not(before), after))
    assert int(popcount[np.bitwise_xor(before, after)].sum(dtype=np.int64)) == k
    assert np.array_equal(chosen, uniform_subset_reference(total_ones, k, 17))
    assert not np.array_equal(before, after)
    print('PASS: reference bit-level selection self-test, exact 1->0 count, reproducibility, no 0->1', flush=True)

@torch.no_grad()
def inject_paper(layers, seed, folder, p=None):
    # Exact reference protocol: sample uniformly from all initially-1 payload bits.
    # Denominator is all INT8 payload bits; no lm_head; no zero-bit resampling is needed.
    restore_clean_qweights(layers)
    folder = safe(folder)
    folder.mkdir(parents=True, exist_ok=True)
    popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    nth_bit = np.zeros((256, 8), dtype=np.uint8)
    for value in range(256):
        ones = [bit for bit in range(8) if value & (1 << bit)]
        nth_bit[value, :len(ones)] = ones

    payloads = []
    total_bits = 0
    total_ones = 0
    for name, m in layers:
        clean = m.clean.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
        ones = int(popcount[clean].sum(dtype=np.int64))
        payloads.append((name, m, clean.size, ones))
        total_bits += int(clean.size) * 8
        total_ones += ones
    assert payloads and total_bits > 0
    target_ber = float(CFG['target_BER'])
    k = (total_bits * 3 + 500) // 1000 if target_ber == 0.003 else int(round(total_bits * target_ber))
    assert k <= total_ones, (k, total_ones)
    chosen = uniform_subset_reference(total_ones, k, seed)
    print('reference_BITS_BEFORE', total_bits, 'ONES', total_ones, 'K', k, 'SEED', seed, flush=True)

    base_ones = 0
    rows = []
    actual = 0
    for name, m, size, layer_ones in payloads:
        original = m.clean.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
        layer_changed = 0
        layer_selected = 0
        cumulative_layer = 0
        flat_qweight = m.qweight.view(torch.int8).reshape(-1)
        for start in range(0, size, 1048576):
            end = min(start + 1048576, size)
            before = original[start:end].copy()
            counts = popcount[before]
            cumulative = np.cumsum(counts, dtype=np.int64)
            global_lo = base_ones + cumulative_layer
            global_hi = base_ones + cumulative_layer + int(cumulative[-1])
            lo = int(np.searchsorted(chosen, global_lo, side='left'))
            hi = int(np.searchsorted(chosen, global_hi, side='left'))
            after = before
            if hi > lo:
                ranks = chosen[lo:hi] - base_ones - cumulative_layer
                byte_ids = np.searchsorted(cumulative, ranks, side='right')
                prior = cumulative[byte_ids] - counts[byte_ids]
                within = (ranks - prior).astype(np.int64)
                bits = nth_bit[before[byte_ids], within]
                mask = np.zeros(before.size, dtype=np.uint8)
                np.bitwise_or.at(mask, byte_ids, np.left_shift(np.uint8(1), bits))
                after = np.bitwise_and(before, np.bitwise_not(mask))
                assert not np.any(np.bitwise_and(np.bitwise_not(before), after))
                changed = int(popcount[np.bitwise_xor(before, after)].sum(dtype=np.int64))
                assert changed == hi - lo
                layer_changed += changed
                layer_selected += hi - lo
            flat_qweight[start:end].copy_(torch.from_numpy(after.view(np.int8)).to(m.qweight.device))
            cumulative_layer += int(cumulative[-1])
        assert cumulative_layer == layer_ones
        assert layer_selected == layer_changed
        assert torch.equal(m.scale.detach().cpu(), m.scale_clean)
        m.refresh()
        rows.append(dict(name=name, shape=list(m.qweight.shape), total_ones=layer_ones,
                         selected_bits=layer_selected, changed_bits=layer_changed))
        base_ones += layer_ones
        actual += layer_changed
    assert base_ones == total_ones and actual == k
    info = dict(
        fault_mode='reference_global_existing_one_bit', target_BER=target_ber,
        BER_definition='actual_changed_bits / total_int8_bits', seed=int(seed),
        fault_direction='1_to_0', selection='uniform subset of all initially-1 INT8 payload bits',
        resample_if_selected_bit_is_zero=False, max_effective_faults_per_weight=None,
        inject_lm_head=False, total_int8_weights=total_bits // 8, total_bits=total_bits,
        total_ones=total_ones, selected_bits=k, effective_1to0_flips=actual,
        effective_0to1_flips=0, actual_BER=actual / total_bits, layers=rows,
    )
    dump(folder / 'fault_manifest.json', info)
    print('reference_FAULT_VERIFIED', json.dumps({k:v for k,v in info.items() if k != 'layers'}, ensure_ascii=False), flush=True)
    return info


def run_v3():
    torch.set_num_threads(4); torch.manual_seed(1234); np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    reference_self_test()
    if CFG.get('test_only'): status('tests_complete'); return
    data,by_subject=load_data()
    old=ROOT/'results_int8/qwen3_8b_w8a16_g128_ber003_v1'
    revision=json.loads((old/'model_revision.json').read_text())['revision']
    kwargs=dict(cache_dir=str(safe('.runtime/hf/hub')),revision=revision,trust_remote_code=False,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(CFG['model_id'],**kwargs); tokenizer.pad_token=tokenizer.eos_token
    prompts=build_prompts(tokenizer,data,by_subject)
    bits=CFG['bits']; label=f'clean_w{bits}a16'
    signature=dict(bits=bits,group_size=128,method='symmetric RTN',target_BER=CFG['target_BER'],BER_definition='actual_changed_bits / total_int8_bits',fault_direction='1_to_0',resample_if_selected_bit_is_zero=False,max_effective_faults_per_weight=None,sampling_scope='global_model',fault_seeds=CFG['fault_seeds'],prompt_hashes={k:digest(v) for k,v in prompts.items()},revision=revision,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),batch_size=CFG['mathqa_batch_size'],mathqa_max_new_tokens=CFG['mathqa_max_new_tokens'],humaneval_max_new_tokens=CFG['max_new_tokens'])
    signature.update(fault_mode='reference_global_existing_one_bit',parameter_error_rate=None,target_BER=CFG['target_BER'],BER_definition='actual_changed_bits / total_int8_bits',resample_if_selected_bit_is_zero=False,inject_lm_head=False,injection_matrix_count=252,compute_dtype='bfloat16',evaluation_tasks=['mathqa','mmlu','humaneval'],selection='uniform subset of all initially-1 INT8 payload bits',max_effective_faults_per_weight=None)
    manifest=OUT/f'manifest_int{bits}.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==signature,'Do not mix changed experiments'
    else: dump(manifest,signature)
    if bits==4 and (OUT/label/'metrics.json').exists(): status('complete',bits=4); return
    if CFG.get('dry_run'):
        status('dry_run_loading_cpu')
        model=AutoModelForCausalLM.from_pretrained(CFG['model_id'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cpu'},**kwargs).eval()
        status('dry_run_quantizing_cpu')
        layers=quantize_cpu(model,8); gc.collect()
        for name,m in layers: m.refresh=lambda: None
        info=inject_paper([(n,m) for n,m in layers if n!='lm_head'],CFG['fault_seeds'][0],OUT/'seed_000/faults')
        dump(OUT/'dry_run_verified.json',info)
        status('dry_run_complete',effective_flips=info['effective_1to0_flips'],target_error_bits=info['target_error_bits'],effective_BER=info['effective_BER'])
        return
    device=CFG['device_index']; torch.cuda.set_device(device)
    assert torch.cuda.mem_get_info()[0]>23*1024**3,'GPU memory changed; retry later'
    status('loading_cpu',bits=bits,gpu=device)
    model=AutoModelForCausalLM.from_pretrained(CFG['model_id'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cpu'},**kwargs).eval()
    status('quantizing_cpu',bits=bits)
    layers=quantize_cpu(model,bits); assert len(layers)==253 and sum(n!='lm_head' for n,m in layers)==252; gc.collect()
    model.to(device)
    for name,m in layers: m.target_device=torch.device('cuda',device); m.refresh()
    status('evaluating_clean',bits=bits,gpu=device,gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3)
    if not CFG.get('dry_run') and not CFG.get('skip_clean_w8a16',False): run_all_benchmarks(model,tokenizer,prompts,label)
    if CFG.get('skip_clean_w8a16',False): status('skipping_clean_w8a16')
    baselines={}
    for tag in ['clean_w4a16','clean_w8a16']:
        p=OUT/tag/'metrics.json'
        if p.exists(): baselines[tag]=json.loads(p.read_text())
    v2=ROOT/'results_int8/qwen3_8b_v2_wer003_mathqa_rationale'
    for tag in ['fp16','bf16']:
        p=v2/tag/'metrics.json'
        if p.exists(): baselines[tag]=json.loads(p.read_text())
    dump(OUT/'baselines.json',baselines)
    if bits==4: status('complete',bits=4); return
    for seed in CFG['fault_seeds']:
        directory=OUT/f'seed_{seed:03d}'
        if not (directory/'metrics.json').exists():
            status('injecting_reference_bitber003',seed=seed,target_BER=CFG['target_BER'])
            info=inject_paper([(n,m) for n,m in layers if n!='lm_head'],seed,directory/'faults')
            if CFG.get('dry_run'):
                dump(OUT/'dry_run_verified.json',info); status('dry_run_complete',seed=seed,effective_flips=info['effective_1to0_flips'],target_error_bits=info['target_error_bits'],effective_BER=info['effective_BER']); return
            head=dict(layers)['lm_head']; assert torch.equal(head.qweight,head.clean)
            run_all_benchmarks(model,tokenizer,prompts,directory.name)
        summary_v6()
    report=summary_v6()
    status('complete',bits=8,fault_runs=len(report['completed_seeds']))
def inject_paper_legacy(layers,seed,folder,p=None):
    p=CFG['parameter_error_rate'] if p is None else p
    assert 0<=p<=1 and all(n!='lm_head' for n,m in layers)
    folder=safe(folder); folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(int(seed)); pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    rows=[]; offset=0; ones=0; attempts=0; actual=0
    for name,m in layers:
        assert m.bits==8 and m.qweight.dtype==torch.int8
        m.qweight.copy_(m.clean); assert torch.equal(m.scale,m.scale_clean)
        c=m.clean.numpy().view(np.uint8).reshape(-1); q=m.qweight.numpy().view(np.uint8).reshape(-1)
        ids_all=[]; bits_all=[]; na=0; nf=0; no=0
        for start in range(0,len(c),1048576):
            end=min(start+1048576,len(c)); raw=c[start:end]
            ids=np.flatnonzero(rng.random(end-start)<p)
            bits=rng.integers(0,8,size=len(ids),dtype=np.uint8)
            masks=np.left_shift(np.uint8(1),bits)
            before=raw[ids]; after=before & np.bitwise_not(masks)
            q[start+ids]=after
            delta=raw ^ q[start:end]
            assert not np.any(q[start:end] & np.bitwise_not(raw))
            assert np.all(pop[delta]<=1)
            assert int(pop[delta].sum())==int(np.count_nonzero(before!=after))
            na+=len(ids); nf+=int(pop[delta].sum()); no+=int(pop[raw].sum())
            ids_all.append(ids.astype(np.int64)+start); bits_all.append(bits)
        np.savez_compressed(folder/(name+'.npz'),selected_local_indices=np.concatenate(ids_all),bit_positions=np.concatenate(bits_all))
        rows.append(dict(tensor_name=name,num_weights=len(c),global_weight_offset=offset,selected_parameters=na,effective_flips=nf,total_ones=no))
        offset+=len(c); attempts+=na; actual+=nf; ones+=no
        assert torch.equal(m.scale,m.scale_clean); m.refresh()
    meta=dict(fault_mode='paper_parameter_attempt',parameter_error_rate=p,selection='independent Bernoulli per parameter',seed=int(seed),fault_direction='1_to_0',resample_if_selected_bit_is_zero=False,inject_lm_head=False,total_int8_weights=offset,total_bits=8*offset,total_ones=ones,ones_fraction=ones/(8*offset),selected_parameters=attempts,zero_bit_attempts=attempts-actual,effective_1to0_flips=actual,effective_0to1_flips=0,effective_BER=actual/(8*offset),layers=rows)
    dump(folder/'fault_manifest.json',meta)
    print('PAPER_FAULT_VERIFIED',json.dumps({k:v for k,v in meta.items() if k!='layers'}),flush=True)
    return meta

def test_paper():
    class Toy:
        bits=8
        def __init__(self,values):
            self.clean=torch.tensor(values,dtype=torch.int8); self.qweight=self.clean.clone()
            self.scale=torch.ones(1); self.scale_clean=self.scale.clone()
        def refresh(self): pass
    z=Toy([0]*256); a=inject_paper([('toy',z)],0,OUT/'paper_tests/zero',p=1)
    assert a['selected_parameters']==256 and a['effective_1to0_flips']==0
    z=Toy([-1]*256); a=inject_paper([('toy',z)],0,OUT/'paper_tests/ones',p=1)
    assert a['effective_1to0_flips']==256
    saved=z.qweight.clone(); inject_paper([('toy',z)],0,OUT/'paper_tests/replay',p=1)
    assert torch.equal(saved,z.qweight)
    inject_paper([('toy',z)],1,OUT/'paper_tests/none',p=0); assert torch.equal(z.clean,z.qweight)
    try: inject_paper([('lm_head',z)],0,OUT/'paper_tests/forbidden',p=1)
    except AssertionError: pass
    else: raise AssertionError('lm_head not excluded')
    dump(OUT/'paper_self_test.json',dict(passed=True,tests=['zero no resampling','one bit per selected weight','seed restore replay','zero probability clean','lm_head exclusion']))


def run_strict_aligned_clean():
    torch.set_num_threads(4)
    torch.manual_seed(1234); np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    status('strict_self_test')
    if 'reference_self_test' in globals():
        reference_self_test()
    data, by_subject = load_data()
    revision = CFG.get('model_revision') or 'main'
    model_source = str(safe(CFG['local_model'])) if CFG.get('local_model') else CFG['model_id']
    kwargs = dict(cache_dir=str(safe('.runtime/hf/hub')), revision=revision, trust_remote_code=False, local_files_only=bool(CFG.get('local_files_only', False)))
    tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    prompts = build_prompts(tokenizer, data, by_subject)
    assert CFG['bits'] == 8
    device = CFG['device_index']
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    assert free > 23 * 1024**3, f'GPU memory changed; free={free/1024**3:.2f} GiB'
    status('loading_cpu', bits=8, gpu=device, free_gib=free/1024**3)
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cpu'}, **kwargs).eval()
    status('quantizing_cpu', bits=8)
    layers = quantize_cpu(model, 8)
    assert len(layers) == 253 and sum(name != 'lm_head' for name, _ in layers) == 252
    gc.collect()
    model.to(device)
    for name, layer in layers:
        layer.target_device = torch.device('cuda', device)
        layer.refresh()
    status('evaluating_clean', bits=8, gpu=device, gpu_allocated_gib=torch.cuda.memory_allocated(device)/1024**3)
    metrics = run_all_benchmarks(model, tokenizer, prompts, 'clean_w8a16')
    dump(OUT/'strict_clean_metrics.json', metrics)
    dump(OUT/'strict_clean_config.json', dict(
        source_worker_sha256=CFG['source_worker_sha256'], model_id=CFG['model_id'], model_revision=revision,
        bits=8, group_size=128, method='symmetric RTN', compute_dtype='bfloat16',
        fault_injection=False, inject_lm_head=False, resample_if_selected_bit_is_zero=False,
        evaluation_protocol='reused original reference worker: rationale MathQA generation + Final answer parser, MMLU original evaluator, original HumanEval sandbox',
        mathqa_batch_size=CFG['mathqa_batch_size'], mathqa_max_new_tokens=CFG['mathqa_max_new_tokens'],
        humaneval_max_new_tokens=CFG['max_new_tokens'], humaneval_timeout=CFG['humaneval_timeout'],
        prompt_hashes={key: digest(value) for key, value in prompts.items()},
    ))
    status('complete', bits=8, metrics=metrics)

EXPECTED_CLEAN_WORKER_SHA256 = '9f184def8e329a5e053859598fd8acc2e961a77812b0ce0b574173cc95c75fc0'


import zipfile

POPCOUNT8 = np.array([int(v).bit_count() for v in range(256)], dtype=np.uint8)
NTH8 = np.zeros((256, 8), dtype=np.uint8)
for _v in range(256):
    _bits = [b for b in range(8) if _v & (1 << b)]
    NTH8[_v, :len(_bits)] = _bits

METHOD = dict(
    name="HBM-v6 SRLR",
    code="copy each INT8 MSB into its LSB before HBM storage; repair mismatched pair to 11 after 1-to-0 faults",
    protected_payload="252 non-lm_head INT8 qweight matrices; lm_head, nn.Embedding modules, scales and biases are outside the fault/protection payload",
    decoder="for each byte, if MSB and LSB differ after HBM read, set both to 1; other bit positions are not corrected",
    logical_storage_overhead=0.0,
)

_BASE_STATUS = status
_ACTIVE_STATUS = {}

def status(stage, **fields):
    _BASE_STATUS(stage, **{**_ACTIVE_STATUS, **fields})

def protection_status(stage, **fields):
    record = {**_ACTIVE_STATUS, **fields, "method": "srlr", "stage": stage,
        "time": time.strftime('%Y-%m-%d %H:%M:%S')}
    dump(OUT / 'srlr' / 'status.json', record)
    status(stage, method="srlr", **fields)

def _raw_bytes(layer):
    return layer.clean.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)

def _srlr_encode(raw):
    return ((raw & np.uint8(0xFE)) | ((raw >> np.uint8(7)) & np.uint8(1))).astype(np.uint8)

def _srlr_decode(damaged):
    fixed = damaged.copy()
    mismatch = ((fixed & np.uint8(1)) != ((fixed >> np.uint8(7)) & np.uint8(1)))
    corrected = int(np.count_nonzero(mismatch))
    fixed[mismatch] |= np.uint8(0x81)
    return fixed, corrected

def uniform_subset_reference(n, k, seed):
    """Archived reference sampling rule: exact-size uniform subset of eligible one-bit ranks."""
    n, k = int(n), int(k)
    if not 0 <= k <= n:
        raise ValueError(f"invalid one-bit sample: k={k}, n={n}")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected = np.unique(rng.integers(0, n, size=k, dtype=np.int64))
    while selected.size < k:
        selected = np.union1d(selected, rng.integers(0, n, size=k-selected.size, dtype=np.int64))
    return selected

def _append_mask_arrays(archive, layer_index, indexes, bit_positions):
    # Stream NPY arrays into the ZIP to avoid a second full-size BytesIO copy.
    for suffix, values in (("indices", indexes), ("bits", bit_positions)):
        info = zipfile.ZipInfo(f"layer_{layer_index:03d}_{suffix}.npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with archive.open(info, mode="w", force_zip64=True) as stream:
            np.save(stream, values, allow_pickle=False)

def _map_one_bit_ranks(values, ranks, chunk_bytes=1_048_576):
    """Map sorted ranks among a byte array's one-bits to (byte index, bit index), bounded scratch memory."""
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or (ranks.size > 1 and np.any(ranks[1:] <= ranks[:-1])):
        raise RuntimeError("one-bit ranks must be a strictly increasing vector")
    indexes = np.empty(ranks.size, dtype=np.int64)
    bits = np.empty(ranks.size, dtype=np.uint8)
    cursor = 0
    seen = 0
    out = 0
    for start in range(0, int(values.size), int(chunk_bytes)):
        end = min(start + int(chunk_bytes), int(values.size))
        block = values[start:end]
        counts = POPCOUNT8[block]
        cumulative = np.cumsum(counts, dtype=np.int64)
        block_ones = int(cumulative[-1]) if cumulative.size else 0
        stop = int(np.searchsorted(ranks, seen + block_ones, side="left"))
        if stop > cursor:
            local_ranks = ranks[cursor:stop] - seen
            local_indexes = np.searchsorted(cumulative, local_ranks, side="right")
            before = cumulative[local_indexes] - counts[local_indexes]
            within = (local_ranks - before).astype(np.uint8, copy=False)
            n = stop - cursor
            indexes[out:out+n] = start + local_indexes
            bits[out:out+n] = NTH8[block[local_indexes], within]
            out += n
            cursor = stop
        seen += block_ones
        if cursor == ranks.size:
            break
    if cursor != ranks.size or out != ranks.size or (ranks.size and int(ranks[-1]) >= seen):
        raise RuntimeError(f"could not map all selected one-bit ranks: mapped={out}/{ranks.size}")
    return indexes, bits

def _file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _commit_mask_archive(temp_path, mask_path):
    new_hash = _file_sha256(temp_path)
    if mask_path.exists():
        old_hash = _file_sha256(mask_path)
        if old_hash != new_hash:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"frozen SRLR mask differs; refusing overwrite: {mask_path}")
        temp_path.unlink()
    else:
        temp_path.replace(mask_path)
    return new_hash

def _dump_frozen(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != obj:
            raise RuntimeError(f"frozen manifest differs; refusing overwrite: {path}")
        return
    dump(path, obj)

def _copy_immutable(source, destination):
    data = Path(source).read_bytes()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != data:
            raise RuntimeError(f"immutable source snapshot differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

def _srlr_manifest(layers, seed, run_dir):
    selected = [(name, layer) for name, layer in layers if name != 'lm_head']
    if len(selected) != 252 or any('embed_tokens' in name for name, _ in selected):
        raise RuntimeError("unexpected injection scope; expected 252 non-head Linear matrices and no embedding module")
    byte_counts = [int(layer.clean.numel()) for _, layer in selected]
    ones_by_layer = []
    for _, layer in selected:
        encoded = _srlr_encode(_raw_bytes(layer))
        ones_by_layer.append(int(POPCOUNT8[encoded].sum(dtype=np.int64)))
        del encoded
    active_bits = 8 * sum(byte_counts)
    target = (active_bits * 3 + 500) // 1000  # exact reference BER=0.003 half-up rule
    if active_bits != int(CFG['reference_total_active_bits']) or target != int(CFG['reference_target_fault_sites']):
        raise RuntimeError(f"SRLR payload/target differs from the frozen reference INT8 protocol: bits={active_bits}, sites={target}")
    total_ones = int(sum(ones_by_layer))
    if target > total_ones:
        raise RuntimeError(f"SRLR payload has too few one-bits: target={target}, eligible={total_ones}")
    selected_ranks = uniform_subset_reference(total_ones, target, seed)
    if selected_ranks.size != target or (selected_ranks.size > 1 and np.any(selected_ranks[1:] <= selected_ranks[:-1])):
        raise RuntimeError("reference-compatible sampler returned a malformed selected-rank array")
    boundaries = np.cumsum(np.asarray(ones_by_layer, dtype=np.int64))
    mask_hash = hashlib.sha256()
    layer_stats = []
    mask_path = run_dir / 'mask_coordinates.npz'
    temp_path = run_dir / 'mask_coordinates.npz.partial'
    total_selected = 0
    with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        lower = 0
        for i, ((name, layer), upper_value) in enumerate(zip(selected, boundaries)):
            upper = int(upper_value)
            lo = int(np.searchsorted(selected_ranks, lower, side='left'))
            hi = int(np.searchsorted(selected_ranks, upper, side='left'))
            local_ranks = selected_ranks[lo:hi] - lower
            encoded = _srlr_encode(_raw_bytes(layer))
            indexes, bit_positions = _map_one_bit_ranks(encoded, local_ranks)
            if indexes.size and np.any(((encoded[indexes] >> bit_positions) & 1) == 0):
                raise RuntimeError(f"selected SRLR coordinate is not a stored one-bit: {name}")
            mask_hash.update(np.asarray([i], dtype='<u4').tobytes())
            mask_hash.update(indexes.astype('<u8', copy=False).tobytes())
            mask_hash.update(bit_positions.astype('u1', copy=False).tobytes())
            _append_mask_arrays(archive, i, indexes, bit_positions)
            layer_stats.append(dict(tensor_name=name, qweight_bytes=byte_counts[i],
                encoded_initial_one_bits=ones_by_layer[i], attempted_sites=int(indexes.size), actual_1to0_flips=int(indexes.size)))
            total_selected += int(indexes.size)
            lower = upper
            del local_ranks, encoded, indexes, bit_positions
    del selected_ranks
    if total_selected != target:
        raise RuntimeError(f"SRLR selected-flip total mismatch: {total_selected} != {target}")
    mask_file_sha256 = _commit_mask_archive(temp_path, mask_path)
    return dict(method=METHOD, seed=int(seed), sampling_protocol="reference uniform subset of existing 1-bits, independently sampled on the SRLR-encoded payload using the same seed IDs 0-9",
        target_BER=0.003, BER_denominator="all 8 payload bits per non-lm_head INT8 qweight byte; lm_head, embeddings, scales and biases excluded",
        total_active_bits=int(active_bits), eligible_encoded_one_bits=total_ones, target_fault_sites=int(target),
        attempted_fault_sites=int(total_selected), effective_1to0_flips=int(total_selected), effective_BER=float(total_selected / active_bits),
        resample_if_selected_bit_is_zero=False, mask_coordinates_file="mask_coordinates.npz", mask_file_sha256=mask_file_sha256,
        coordinate_sha256=mask_hash.hexdigest(), layer_count=len(selected), layers=layer_stats)

def _validate_coordinates(indexes, bit_positions, maximum_bit, layer_index):
    if indexes.ndim != 1 or bit_positions.ndim != 1 or indexes.size != bit_positions.size:
        raise RuntimeError(f"malformed coordinate arrays in layer {layer_index}")
    if indexes.size and (np.any(indexes[1:] < indexes[:-1]) or np.any(bit_positions > maximum_bit)):
        raise RuntimeError(f"invalid frozen coordinates in layer {layer_index}")
    if indexes.size > 1:
        same = indexes[1:] == indexes[:-1]
        if np.any(bit_positions[1:][same] <= bit_positions[:-1][same]):
            raise RuntimeError(f"duplicate frozen coordinates in layer {layer_index}")

def _apply_srlr_mask(layers, run_dir, expected):
    selected = [(name, layer) for name, layer in layers if name != 'lm_head']
    if len(selected) != 252 or any('embed_tokens' in name for name, _ in selected):
        raise RuntimeError("unexpected SRLR injection scope")
    path = run_dir / 'mask_coordinates.npz'
    if _file_sha256(path) != expected['mask_file_sha256']:
        raise RuntimeError('frozen SRLR coordinate-file SHA256 mismatch')
    mask_hash = hashlib.sha256()
    total = 0
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(selected) * 2:
            raise RuntimeError(f"frozen SRLR layer count mismatch: {path}")
        for i, (name, layer) in enumerate(selected):
            ik, bk = f'layer_{i:03d}_indices', f'layer_{i:03d}_bits'
            if ik not in archive or bk not in archive:
                raise RuntimeError(f"missing frozen SRLR coordinates for layer {i}")
            indexes = archive[ik]
            bit_positions = archive[bk]
            _validate_coordinates(indexes, bit_positions, 7, i)
            mask_hash.update(np.asarray([i], dtype='<u4').tobytes())
            mask_hash.update(indexes.astype('<u8', copy=False).tobytes())
            mask_hash.update(bit_positions.astype('u1', copy=False).tobytes())
            encoded = _srlr_encode(_raw_bytes(layer))
            if indexes.size and (np.any(indexes >= encoded.size) or np.any(((encoded[indexes] >> bit_positions) & 1) == 0)):
                raise RuntimeError(f"frozen SRLR mask no longer points to stored one-bits: {name}")
            if indexes.size:
                encoded[indexes] &= np.bitwise_not(np.left_shift(np.uint8(1), bit_positions))
            decoded, corrected = _srlr_decode(encoded)
            layer.qweight.copy_(torch.from_numpy(decoded.view(np.int8).reshape(layer.qweight.shape)).to(layer.qweight.device))
            layer.refresh()
            total += int(indexes.size)
            del indexes, bit_positions, encoded, decoded
    if mask_hash.hexdigest() != expected['coordinate_sha256'] or total != int(expected['target_fault_sites']):
        raise RuntimeError('frozen SRLR mask digest/count mismatch')
    return total

def protection_self_tests():
    sample = np.array([0, 1, 127, 128, 255], dtype=np.uint8)
    encoded = _srlr_encode(sample)
    assert np.array_equal(_srlr_decode(encoded)[0], encoded)
    for original in sample:
        protected = _srlr_encode(np.array([original], dtype=np.uint8))
        if protected[0] & 1:
            for bit in (0, 7):
                damaged = protected.copy()
                damaged[0] &= np.uint8(~(1 << bit) & 255)
                restored, _ = _srlr_decode(damaged)
                assert restored[0] & 0x81 == 0x81
    # Selected coordinate counts agree with the archived reference exact-size sampler.
    ranks = uniform_subset_reference(100_000, 300, 17)
    assert ranks.size == 300 and np.all(ranks[1:] > ranks[:-1])
    values = np.array([0b00100101, 0b10000001, 0b00000000], dtype=np.uint8)
    mapped_i, mapped_b = _map_one_bit_ranks(values, np.array([0, 1, 2, 3], dtype=np.int64), chunk_bytes=1)
    assert list(zip(mapped_i.tolist(), mapped_b.tolist())) == [(0, 0), (0, 2), (0, 5), (1, 0)]
    print('PASS: SRLR encoding/1->0 repair, reference sampler and chunked coordinate mapping', flush=True)

def _load_clean_reference():
    clean_root = safe(Path('results_int8') / CFG['clean_int8_result_name'])
    status_path = clean_root / 'status.json'
    config_path = clean_root / 'strict_clean_config.json'
    metrics_path = clean_root / 'strict_clean_metrics.json'
    for path in (status_path, config_path, metrics_path):
        if not path.is_file():
            raise RuntimeError(f"missing result-linked clean INT8 reference: {path}")
    reference_status = json.loads(status_path.read_text(encoding='utf-8'))
    reference_config = json.loads(config_path.read_text(encoding='utf-8'))
    reference_metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if reference_status.get('stage') != 'complete' or reference_config.get('fault_injection') is not False:
        raise RuntimeError('clean INT8 reference is not a completed no-fault run')
    expected = dict(bits=8, group_size=128, method='symmetric RTN', compute_dtype='bfloat16', inject_lm_head=False)
    for key, value in expected.items():
        if reference_config.get(key) != value:
            raise RuntimeError(f"clean INT8 reference mismatch for {key}: {reference_config.get(key)!r}")
    scores = {key: float(reference_metrics[key]['accuracy']) * 100 for key in ('mmlu', 'humaneval')}
    if abs(scores['mmlu'] - 72.20) > 1e-9 or abs(scores['humaneval'] - (137 / 164 * 100)) > 1e-9:
        raise RuntimeError(f"clean INT8 comparison scores differ from frozen reference: {scores}")
    return clean_root, reference_config, reference_metrics, scores

def _update_mask_plan(records, phase):
    plan = dict(experiment=CFG['result_name'], phase=phase, protection='SRLR', fault_seeds=[int(s) for s in CFG['fault_seeds']],
        frozen_seed_count=len(records), required_seed_count=len(CFG['fault_seeds']),
        mask_file='srlr/seed_###/mask_coordinates.npz', mask_hash='fault_manifest.json:mask_file_sha256 and coordinate_sha256',
        per_seed=records, inference_starts_only_after_all_masks_frozen=True)
    path = OUT / 'mask_plan.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old.get('experiment') != CFG['result_name'] or old.get('protection') != 'SRLR' or old.get('fault_seeds') != plan['fault_seeds']:
            raise RuntimeError('existing mask plan belongs to a different experiment; refusing to reuse')
    dump(path, plan)

def _summarize_dataset(method_dir, dataset, clean_scores):
    metric_name = f'{dataset}_metrics.json'
    records = []
    for seed in CFG['fault_seeds']:
        path = method_dir / f'seed_{int(seed):03d}' / metric_name
        if path.exists():
            metric = json.loads(path.read_text(encoding='utf-8'))
            records.append(dict(seed=int(seed), correct=int(metric['correct']), total=int(metric['total']),
                accuracy_pct=float(metric['accuracy']) * 100, effective_BER=float(metric['effective_BER']),
                coordinate_sha256=metric['coordinate_sha256']))
    values = [r['accuracy_pct'] for r in records]
    n = len(values)
    mean = float(np.mean(values)) if n else None
    std = float(np.std(values, ddof=1)) if n > 1 else None
    half = float(student_t.ppf(.975, n-1) * std / math.sqrt(n)) if n > 1 and std is not None else None
    report = dict(method='srlr', dataset=dataset, completed_seeds=[r['seed'] for r in records], n=n,
        clean_int8_baseline_pct=clean_scores[dataset], fault_mean_pct=mean, std_pp=std,
        ci95_low_pct=None if half is None else mean-half, ci95_high_pct=None if half is None else mean+half,
        fault_minus_clean_pp=None if mean is None else mean-clean_scores[dataset], per_seed=records,
        std_definition='sample standard deviation (ddof=1), percentage points')
    dump(method_dir / f'summary_{dataset}.json', report)
    pd.DataFrame(records).to_csv(method_dir / f'runs_{dataset}.csv', index=False)
    return report

def _clean_srlr_control(model, tokenizer, layers, prompts, prompt_hashes, method_dir, clean_scores):
    control = method_dir / 'protected_clean_control'
    control.mkdir(parents=True, exist_ok=True)
    results = {}
    restore_clean_qweights(layers)
    try:
        for name, layer in _quant_layers(layers):
            raw = _raw_bytes(layer)
            encoded = _srlr_encode(raw)
            layer.qweight.copy_(torch.from_numpy(encoded.view(np.int8).reshape(layer.qweight.shape)).to(layer.qweight.device))
            layer.refresh()
            del raw, encoded
        for dataset in ('mmlu', 'humaneval'):
            metric_path = control / f'{dataset}_metrics.json'
            _ACTIVE_STATUS.clear(); _ACTIVE_STATUS.update(dict(method='srlr', seed=None, dataset=dataset, phase='protected_clean_control'))
            if metric_path.exists():
                metric = json.loads(metric_path.read_text(encoding='utf-8'))
            else:
                protection_status('evaluating_clean_control', dataset=dataset, completed=0, total=len(prompts[dataset]))
                result = evaluate_dataset(model, tokenizer, dataset, prompts[dataset], control)
                metric = dict(method='srlr', seed=None, dataset=dataset, control='SRLR encoded weights without injected faults',
                    **result, accuracy_pct=100*float(result['accuracy']), clean_int8_baseline_pct=clean_scores[dataset],
                    accuracy_delta_vs_clean_pp=100*float(result['accuracy'])-clean_scores[dataset], prompt_sha256=prompt_hashes[dataset])
                dump(metric_path, metric)
            results[dataset] = metric
    finally:
        restore_clean_qweights(layers)
    dump(control / 'metrics.json', results)
    return results

def run_int8_srlr():
    torch.set_num_threads(4); torch.manual_seed(1234); np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    protection_self_tests()
    if CFG.get('method_order') != ['srlr'] or CFG.get('target_BER') != 0.003 or CFG.get('fault_direction') != '1_to_0':
        raise RuntimeError('this frozen worker only accepts the configured SRLR / BER=0.003 / 1-to-0 experiment')
    if CFG.get('inject_lm_head') is not False or CFG.get('inject_embeddings') is not False:
        raise RuntimeError('lm_head and embeddings must remain outside the fault scope')
    if [int(s) for s in CFG['fault_seeds']] != list(range(10)):
        raise RuntimeError('expected the ten paired seed IDs 0-9')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_path = Path(sys.argv[1]).resolve()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if CFG.get('reference_clean_worker_sha256') != EXPECTED_CLEAN_WORKER_SHA256:
        raise RuntimeError('config does not name the verified canonical clean INT8 worker')
    clean_root, reference_config, reference_metrics, clean_scores = _load_clean_reference()
    selected_methods = ['srlr']
    experiment_manifest = dict(experiment='Qwen3-8B W8A16 INT8 BER=0.003 SRLR protection',
        worker_sha256=source_hash, config_sha256=config_hash, source_clean_worker_sha256=CFG['reference_clean_worker_sha256'],
        source_reference_worker_sha256=CFG['source_worker_sha256'], model_id=CFG['model_id'], model_revision=CFG['model_revision'],
        bits=8, group_size=128, quantization='symmetric RTN, q in [-127,127]', compute_dtype='bfloat16',
        injection_scope='252 non-lm_head INT8 Linear qweight matrices; embeddings, lm_head, scales, and biases excluded',
        protection_scope=METHOD['protected_payload'], fault_direction='1_to_0', target_BER=0.003,
        total_active_bits=int(CFG['reference_total_active_bits']), target_fault_sites=int(CFG['reference_target_fault_sites']),
        fault_seeds=[int(s) for s in CFG['fault_seeds']],
        seed_policy='seed IDs 0-9 paired by ID only; independent deterministic masks sampled uniformly from existing 1-bits in each SRLR-encoded payload; coordinate equality to the unprotected masks is not claimed',
        mask_policy='all ten exact-size 0.003 BER masks are frozen before any inference; the same SRLR seed mask is replayed for MMLU and HumanEval',
        evaluation_order=['mmlu: seeds 0-9', 'humaneval: seeds 0-9', 'protected clean controls: mmlu then humaneval'],
        evaluation='same 1,000 fixed 5-shot MMLU and 164 HumanEval prompts/evaluators as the strict-aligned clean INT8 worker',
        clean_int8_baseline_pct=clean_scores, clean_int8_prompt_hashes=reference_config['prompt_hashes'],
        methods={m:METHOD for m in selected_methods})
    manifest_path = OUT / 'experiment_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != experiment_manifest:
            raise RuntimeError('existing experiment manifest differs; refusing to mix configurations')
    elif any(OUT.iterdir()):
        raise RuntimeError(f'non-empty output without matching experiment manifest: {OUT}')
    else:
        dump(manifest_path, experiment_manifest)
    _copy_immutable(Path(__file__), OUT / 'worker_snapshot.py')
    _copy_immutable(config_path, OUT / 'config_snapshot.json')
    _ACTIVE_STATUS.clear(); _ACTIVE_STATUS.update(dict(method='srlr'))
    protection_status('preflight', fault_seeds=CFG['fault_seeds'], datasets=['mmlu', 'humaneval'])
    data, by_subject = load_data()
    reference_data_path = clean_root / 'data_manifest.json'
    if not reference_data_path.is_file():
        raise RuntimeError(f'missing clean INT8 data manifest: {reference_data_path}')
    data_manifest = json.loads((OUT / 'data_manifest.json').read_text(encoding='utf-8'))
    reference_data = json.loads(reference_data_path.read_text(encoding='utf-8'))
    for key in ('mmlu_fixed_1000', 'mmlu_fixed_5shot', 'humaneval'):
        if data_manifest[key] != reference_data[key]:
            raise RuntimeError(f'fixed dataset mismatch against clean INT8: {key}')
    kwargs = dict(cache_dir=str(safe('.runtime/hf/hub')), revision=CFG['model_revision'], trust_remote_code=False, local_files_only=True)
    model_source = str(safe(CFG['local_model']))
    tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs); tokenizer.pad_token = tokenizer.eos_token
    prompts = build_prompts(tokenizer, data, by_subject)
    prompt_hashes = {key: digest(prompts[key]) for key in ('mmlu', 'humaneval')}
    for key in ('mmlu', 'humaneval'):
        if prompt_hashes[key] != reference_config['prompt_hashes'][key]:
            raise RuntimeError(f'prompt/evaluator mismatch against clean INT8: {key}')
    if len(prompts['mmlu']) != 1000 or len(prompts['humaneval']) != 164:
        raise RuntimeError('fixed evaluation task counts differ from the clean INT8 reference')
    prompt_manifest = dict(counts={key: len(prompts[key]) for key in ('mmlu', 'humaneval')},
        sha256=prompt_hashes, reference_clean_int8_prompt_hashes=reference_config['prompt_hashes'],
        evaluators=dict(mmlu='strict-aligned choice log-probability', humaneval='strict-aligned deterministic generation + sandbox unit tests'))
    dump(OUT / 'protection_prompt_manifest.json', prompt_manifest)
    device = int(CFG['device_index']); torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free <= 23 * 1024**3:
        raise RuntimeError(f'insufficient free GPU memory for the aligned INT8 worker: {free/1024**3:.2f} GiB')
    status('loading_model_cpu', gpu=device, free_gib=free/1024**3)
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'cpu'}, **kwargs).eval()
    status('quantizing_cpu', bits=8, group_size=128, method='symmetric RTN')
    layers = quantize_cpu(model, 8)
    if len(layers) != 253 or sum(name != 'lm_head' for name, _ in layers) != 252:
        raise RuntimeError(f'unexpected quantized Linear module inventory: {len(layers)}')
    if any('embed_tokens' in name for name, _ in layers):
        raise RuntimeError('embedding unexpectedly appeared in quantized Linear module list')
    if not any(name == 'lm_head' for name, _ in layers):
        raise RuntimeError('expected lm_head in the clean quantized model so it can be explicitly excluded from faults')
    gc.collect(); model.to(device)
    for _, layer in layers:
        layer.target_device = torch.device('cuda', device); layer.refresh()
    status('ready', gpu=device, gpu_allocated_gib=torch.cuda.memory_allocated(device)/1024**3,
        mmlu_count=len(prompts['mmlu']), humaneval_count=len(prompts['humaneval']), prompt_hashes=prompt_hashes)
    method_dir = OUT / 'srlr'; method_dir.mkdir(parents=True, exist_ok=True)
    protection_status('freezing_masks', fault_seeds=CFG['fault_seeds'])
    mask_records = []
    for seed_value in CFG['fault_seeds']:
        seed = int(seed_value); run_dir = method_dir / f'seed_{seed:03d}'; run_dir.mkdir(parents=True, exist_ok=True)
        restore_clean_qweights(layers)
        fault = _srlr_manifest(layers, seed, run_dir)
        _dump_frozen(run_dir / 'fault_manifest.json', fault)
        mask_records.append(dict(seed=seed, mask_file=f'srlr/seed_{seed:03d}/mask_coordinates.npz',
            mask_file_sha256=fault['mask_file_sha256'], coordinate_sha256=fault['coordinate_sha256'],
            target_fault_sites=fault['target_fault_sites'], effective_1to0_flips=fault['effective_1to0_flips'], effective_BER=fault['effective_BER']))
        _update_mask_plan(mask_records, 'freezing')
        protection_status('mask_frozen', seed=seed, frozen_seed_count=len(mask_records), required_seed_count=10,
            effective_BER=fault['effective_BER'], mask_file_sha256=fault['mask_file_sha256'])
    _update_mask_plan(mask_records, 'all_masks_frozen')
    protection_status('all_masks_frozen', frozen_seed_count=10, required_seed_count=10)
    dataset_summaries = {}
    # Complete all ten MMLU runs first, then all ten HumanEval runs.
    for dataset in ('mmlu', 'humaneval'):
        for seed_value in CFG['fault_seeds']:
            seed = int(seed_value); run_dir = method_dir / f'seed_{seed:03d}'
            metric_path = run_dir / f'{dataset}_metrics.json'
            if metric_path.exists():
                continue
            restore_clean_qweights(layers)
            fault = json.loads((run_dir / 'fault_manifest.json').read_text(encoding='utf-8'))
            _ACTIVE_STATUS.clear(); _ACTIVE_STATUS.update(dict(method='srlr', seed=seed, dataset=dataset,
                phase='fault_injection_and_evaluation', completed_seeds=_summarize_dataset(method_dir, dataset, clean_scores)['completed_seeds']))
            protection_status('injecting', dataset=dataset, effective_BER=fault['effective_BER'],
                mask_sha256=fault['coordinate_sha256'])
            applied = _apply_srlr_mask(layers, run_dir, fault)
            if applied != int(fault['target_fault_sites']):
                raise RuntimeError(f'applied site count differs for seed {seed}: {applied}')
            try:
                protection_status('evaluating', dataset=dataset, completed=0, total=len(prompts[dataset]),
                    effective_BER=fault['effective_BER'], mask_sha256=fault['coordinate_sha256'])
                result = evaluate_dataset(model, tokenizer, dataset, prompts[dataset], run_dir)
                metric = dict(method='srlr', seed=seed, dataset=dataset, **result,
                    accuracy_pct=100*float(result['accuracy']), clean_int8_baseline_pct=clean_scores[dataset],
                    accuracy_delta_vs_clean_pp=100*float(result['accuracy'])-clean_scores[dataset],
                    target_BER=0.003, effective_BER=fault['effective_BER'], target_fault_sites=fault['target_fault_sites'],
                    actual_1to0_flips=fault['effective_1to0_flips'], coordinate_sha256=fault['coordinate_sha256'],
                    mask_file_sha256=fault['mask_file_sha256'], prompt_sha256=prompt_hashes[dataset])
                dump(metric_path, metric)
            finally:
                restore_clean_qweights(layers)
            summary = _summarize_dataset(method_dir, dataset, clean_scores)
            protection_status('seed_dataset_complete', dataset=dataset, seed=seed,
                completed_seeds=summary['completed_seeds'], completed_count=summary['n'], required_count=10,
                correct=metric['correct'], total=metric['total'], accuracy_pct=metric['accuracy_pct'],
                mean_accuracy_pct=summary['fault_mean_pct'])
        dataset_summaries[dataset] = _summarize_dataset(method_dir, dataset, clean_scores)
        protection_status('dataset_complete', dataset=dataset, completed_count=dataset_summaries[dataset]['n'],
            required_count=10, mean_accuracy_pct=dataset_summaries[dataset]['fault_mean_pct'],
            std_pp=dataset_summaries[dataset]['std_pp'])
    clean_control = _clean_srlr_control(model, tokenizer, layers, prompts, prompt_hashes, method_dir, clean_scores)
    dump(method_dir / 'summary.json', dict(method='srlr', datasets=dataset_summaries,
        protected_clean_control={key: dict(correct=value['correct'], total=value['total'], accuracy_pct=value['accuracy_pct'])
            for key, value in clean_control.items()}))
    dump(OUT / 'final_summary.json', dict(method='srlr', datasets=dataset_summaries,
        protected_clean_control={key: dict(correct=value['correct'], total=value['total'], accuracy_pct=value['accuracy_pct'])
            for key, value in clean_control.items()}))
    _ACTIVE_STATUS.clear()
    protection_status('complete', datasets={key: dict(completed=r['n'], total=10, mean_accuracy_pct=r['fault_mean_pct'], std_pp=r['std_pp'])
        for key, r in dataset_summaries.items()}, protected_clean_control={key: value['accuracy_pct'] for key, value in clean_control.items()})

try:
    run_int8_srlr()
except BaseException as exc:
    status('failed', error=repr(exc)); traceback.print_exc(); raise
