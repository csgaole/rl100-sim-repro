"""A100-only runtime optimizations; no checkpoint or loss-math changes."""
import ast,inspect,textwrap,types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

def tensor_metrics(method):
    """Return detached CUDA metrics, deferring CPU synchronization to update end."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
    class DeferredMetrics(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='item' and not node.args and not node.keywords:
                node.func.attr='detach'
            return node
    tree=DeferredMetrics().visit(tree);ast.fix_missing_locations(tree)
    scope=dict(method.__globals__);exec(compile(tree,inspect.getsourcefile(method),'exec'),scope)
    return scope[method.__name__]

def install_deferred_metrics(policy):
    policy.compute_loss=types.MethodType(tensor_metrics(type(policy).compute_loss),policy)
    encoder=policy.obs_encoder
    encoder.Recon_VIB_loss=types.MethodType(tensor_metrics(type(encoder).Recon_VIB_loss),encoder)

def host_batch(data,ids,pin=True):
    ep=ids//60;t=ids%60;previous=np.maximum(t-1,0)
    images=np.stack([data['rgb'][ep,previous],data['rgb'][ep,t]],1)
    result={c:np.ascontiguousarray(images[:,:,k]) for k,c in enumerate(['global','left_wrist','right_wrist'])}
    result['agent_pos']=np.stack([data['state'][ep,previous],data['state'][ep,t]],1)
    a=data['action'][ep,t];result['action']=np.concatenate([a[:,:1],a],1)
    return {k:torch.from_numpy(v).pin_memory() if pin else torch.from_numpy(v) for k,v in result.items()}

def prefetch_batches(data,ids,microbatch):
    """One host worker, one next-batch copy stream; no CUDA RNG in worker."""
    chunks=[ids[j:j+microbatch] for j in range(0,len(ids),microbatch)]
    if not chunks:return
    stream=torch.cuda.Stream()
    def transfer(cpu):
        with torch.cuda.stream(stream):
            gpu={k:v.to('cuda',non_blocking=True) for k,v in cpu.items()}
            for k in ['global','left_wrist','right_wrist']:gpu[k]=gpu[k].float().div_(255)
            ready=torch.cuda.Event();ready.record(stream)
        return gpu,ready,cpu
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(host_batch,data,chunks[0]);pending=transfer(future.result())
        for i,chunk in enumerate(chunks):
            # Enqueue next transfer before yielding current batch for compute.
            future=pool.submit(host_batch,data,chunks[i+1]) if i+1<len(chunks) else None
            gpu,event,cpu=pending;torch.cuda.current_stream().wait_event(event)
            for value in gpu.values():value.record_stream(torch.cuda.current_stream())
            nxt=transfer(future.result()) if future is not None else None
            yield {'obs':{k:v for k,v in gpu.items() if k!='action'},'action':gpu['action']},len(chunk)
            # Pinned CPU buffers remain alive through the copy event.
            pending=nxt

def gradient_health(model):
    flags=[];squares=[]
    for p in model.parameters():
        if p.grad is not None:flags.append(torch.isfinite(p.grad).all())
    for p in model.obs_encoder.parameters():
        if p.grad is not None:squares.append(p.grad.detach().square().sum())
    packed=torch.stack([torch.stack(flags).all().float(),torch.stack(squares).sum().sqrt()]).cpu().tolist()
    if not packed[0]:raise RuntimeError('Non-finite gradients')
    return packed[1]
