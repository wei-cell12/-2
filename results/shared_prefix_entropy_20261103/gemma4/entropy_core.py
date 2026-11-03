"""Shared pressured-history entropy comparison; offline, no sampling."""
import math
import numpy as np


def aligned_slice(prompt_length, n_output):
    """Logit at P-1+k predicts y[k], given prompt+y[:k], never y[k]."""
    if prompt_length < 1 or n_output < 1:
        raise ValueError('Nonempty prompt and output required')
    return slice(prompt_length-1, prompt_length-1+n_output)


def log_softmax(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not x.shape[0] or not x.shape[1]:
        raise ValueError('Expected nonempty [tokens,vocabulary] logits')
    if np.isnan(x).any() or np.isposinf(x).any() or np.isneginf(x).all(-1).any():
        raise ValueError('Invalid or fully masked distribution')
    a=x-x.max(-1,keepdims=True)
    return a-np.log(np.exp(a).sum(-1,keepdims=True))


def signals_numpy(pressure,neutral,actual):
    """Full raw-vocabulary distributions. Entropies/JSD use natural logs."""
    a,b=log_softmax(pressure),log_softmax(neutral)
    y=np.asarray(actual,dtype=np.int64)
    if a.shape!=b.shape or y.shape!=(a.shape[0],) or (y<0).any() or (y>=a.shape[1]).any():
        raise ValueError('Inconsistent vocabulary, trajectory, or logits shape')
    p,q=np.exp(a),np.exp(b)
    def product(prob,lp):
        z=np.zeros_like(prob)
        np.multiply(prob,lp,out=z,where=prob>0)
        return z
    mix=np.logaddexp(a,b)-math.log(2)
    da=np.zeros_like(a);db=np.zeros_like(b)
    np.subtract(a,mix,out=da,where=p>0)
    np.subtract(b,mix,out=db,where=q>0)
    hp,h0=-product(p,a).sum(-1),-product(q,b).sum(-1)
    i=np.arange(len(y))
    return dict(H_pressure=hp,H_neutral=h0,delta_H=hp-h0,
        JSD=.5*(product(p,da).sum(-1)+product(q,db).sum(-1)),
        delta_logp=a[i,y]-b[i,y],logp_pressure=a[i,y],logp_neutral=b[i,y],
        top1_pressure=a.argmax(-1),top1_neutral=b.argmax(-1),
        rank_pressure=(a>a[i,y,None]).sum(-1)+1,
        rank_neutral=(b>b[i,y,None]).sum(-1)+1)


def signals_torch(torch,pressure,neutral,actual,chunk=32):
    """Same definitions, FP32 softmax/reductions, bounded temporary memory."""
    if chunk<1 or pressure.ndim!=2 or pressure.shape!=neutral.shape or len(actual)!=pressure.shape[0]:
        raise ValueError('Inconsistent paired logits')
    out={k:[] for k in ['H_pressure','H_neutral','delta_H','JSD','delta_logp',
        'logp_pressure','logp_neutral','top1_pressure','top1_neutral','rank_pressure','rank_neutral']}
    for start in range(0,len(actual),chunk):
        x=pressure[start:start+chunk].float()
        z=neutral[start:start+chunk].to(x.device).float()
        for v in [x,z]:
            if torch.isnan(v).any() or torch.isposinf(v).any() or torch.isneginf(v).all(-1).any():
                raise ValueError('Invalid raw logits')
        a,b=torch.log_softmax(x,-1),torch.log_softmax(z,-1)
        p,q=a.exp(),b.exp();mix=torch.logaddexp(a,b)-math.log(2)
        h1=-(p*torch.where(p>0,a,0)).sum(-1)
        h0=-(q*torch.where(q>0,b,0)).sum(-1)
        js=.5*((p*torch.where(p>0,a-mix,0)).sum(-1)+(q*torch.where(q>0,b-mix,0)).sum(-1))
        y=torch.tensor(actual[start:start+chunk],device=a.device,dtype=torch.long)[:,None]
        la,lb=a.gather(1,y)[:,0],b.gather(1,y)[:,0]
        vals=dict(H_pressure=h1,H_neutral=h0,delta_H=h1-h0,JSD=js,
            delta_logp=la-lb,logp_pressure=la,logp_neutral=lb,
            top1_pressure=a.argmax(-1),top1_neutral=b.argmax(-1),
            rank_pressure=(a>a.gather(1,y)).sum(-1)+1,
            rank_neutral=(b>b.gather(1,y)).sum(-1)+1)
        for k,v in vals.items():out[k].extend(v.detach().cpu().tolist())
    n=len(actual)
    if any(len(v)!=n for v in out.values()):raise AssertionError('Lost token rows')
    for k in ['H_pressure','H_neutral','delta_H','JSD','delta_logp']:
        if not np.isfinite(out[k]).all():raise ValueError('Nonfinite '+k)
    if min(out['JSD']) < -1e-5 or max(out['JSD'])>math.log(2)+1e-5:
        raise ValueError('JSD out of range')
    for k in ['H_pressure','H_neutral']:
        if min(out[k]) < -1e-5 or max(out[k])>math.log(pressure.shape[-1])+1e-5:
            raise ValueError('Entropy out of range')
    return out
