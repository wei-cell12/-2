"""Meaningful CPU property tests, before loading any model."""
import json
import numpy as np
import torch
from intervention import redistribute
from references import construct

def main():
    g=torch.Generator().manual_seed(119)
    tested=[]
    for lam in [0.,1.]:
        for force_cap in [False,True]:
            p=torch.softmax(torch.randn(8,53,generator=g),-1)
            ii=torch.arange(3,43);hs=torch.tensor([1,3,6]);mask=torch.zeros(3,40,dtype=torch.bool);mask[:,:8]=True
            if force_cap:
                p[hs[:,None],ii[:8][None,:]]*=100
                p=p/p.sum(-1,keepdim=True)
                r=torch.full((3,40),.3)
            else:r=torch.softmax(torch.randn(3,40,generator=g),-1)
            out,stats=redistribute(torch,p,ii,hs,mask,r,lam)
            assert torch.allclose(out.sum(-1),p.sum(-1),atol=1e-6)
            assert torch.equal(out[:,torch.cat([torch.arange(3),torch.arange(43,53)])],p[:,torch.cat([torch.arange(3),torch.arange(43,53)])])
            assert torch.equal(out[[0,2,4,5,7]],p[[0,2,4,5,7]])
            assert torch.allclose(out[hs][:,ii].sum(-1),p[hs][:,ii].sum(-1),atol=1e-6)
            if lam==0:assert torch.equal(out,p)
            else:
                assert (stats[:,1]>=stats[:,0]-1e-6).all()
                before=p[hs][:,ii];after=out[hs][:,ii]
                assert (after[:,8:].sum(-1)>=.05*before[:,8:].sum(-1)-1e-6).all()
            tested.append(dict(lambda_=lam,force_cap=force_cap,clamped=int(stats[:,7].sum())))
    # Static references and random controls must be deterministic and disjoint.
    import tempfile
    rng=np.random.default_rng(100)
    d={s:rng.random((16,24,45),dtype=np.float32) for s in ['think','answer']};d.update(n_img=45,grid_hw=np.array([5,9]))
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        a,b=Path(tmp)/'a.npz',Path(tmp)/'b.npz'
        m=construct('CPU-TEST',d,a);construct('CPU-TEST',d,b)
        x,y=np.load(a),np.load(b)
        assert all(np.array_equal(x[k],y[k]) for k in x.files)
        assert m['heads_count']==96 and m['E_size']==9
        assert not (x['E_mask']&x['random_mask']).any()
        assert np.allclose((x['E_r']*x['E_mask']).sum(-1),x['random_r'].sum(-1))
    print(json.dumps(dict(ok=True,properties=tested,random_reference_checks=True,torch=torch.__version__)))
if __name__=='__main__':main()
