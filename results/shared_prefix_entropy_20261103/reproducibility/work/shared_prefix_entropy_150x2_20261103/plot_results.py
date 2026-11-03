import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent
P=json.loads((W/'plot_payload.json').read_text());rng=np.random.default_rng(42)
fig,axes=plt.subplots(2,3,figsize=(15,8.4),sharey=True,layout='constrained')
for row,model in enumerate(['qwen38','gemma4']):
 for col,ds in enumerate(['ClockQA','PathVQA','PENDULUM']):
  ax=axes[row,col];cs=[c for c in P['cases'] if c['model']==model and c['dataset']==ds]
  for ix,group in enumerate(['flip','hold']):
   ids=sorted({c['id'] for c in cs if c['group']==group});counts=[]
   jitter={iid:float(rng.uniform(-.065,.065)) for iid in ids}
   for cond,off,color,label in [('belief',-.11,'#d24d3e','User wrong'),('belief_right',.11,'#2478b7','User right')]:
    vv=[c for c in cs if c['group']==group and c['condition']==cond and c['metrics']['after_opening']['n']];vals=[c['metrics']['after_opening']['means']['delta_H'] for c in vv];counts.append(len(vals))
    ax.scatter([ix+off+jitter[c['id']] for c in vv],vals,color=color,alpha=.55,s=22,label=label if ix==0 else None,zorder=3)
    if vals:ax.plot([ix+off-.07,ix+off+.07],[np.mean(vals)]*2,c=color,lw=3,zorder=4)
   ax.text(ix,.025,'n='+str(counts[0])+'/'+str(counts[1]),ha='center',fontsize=9)
  ax.set_title(('Qwen3.8-27B' if model=='qwen38' else 'Gemma4-31B-it')+' | '+ds);ax.set_xticks([0,1],['Neutral correct → wrong','Neutral correct → correct'],fontsize=9);ax.axhline(0,c='black',lw=.7);ax.grid(axis='y',alpha=.18);ax.set_xlim(-.4,1.4)
  if col==0:ax.set_ylabel('Mean thinking entropy difference (nats)')
axes[0,0].legend(fontsize=9,loc='lower right');fig.suptitle('Shared-history replay | exclude first paragraph and at least64output tokens\nPoints: individual questions; bars: question-weighted means; n: user-wrong / user-right trajectories with remaining thought',fontsize=12)
fig.savefig(W/'cohort_overview.png',dpi=160);plt.close(fig)
# Difference of trajectory means, not a token-by-token semantic alignment across responses.
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained',sharey=True)
for ax,ds in zip(axes,['ClockQA','PathVQA','PENDULUM']):
 for k,model in enumerate(['qwen38','gemma4']):
  for gi,g in enumerate(['flip','hold']):
   vals=[c['metrics']['after64']['means']['delta_H'] for c in P['cases'] if c['model']==model and c['dataset']==ds and c['group']==g and c['condition']=='belief' and c['metrics']['after64']['n']]
   x=k*3+gi
   ax.scatter(np.full(len(vals),x)+rng.uniform(-.12,.12,len(vals)),vals,c=('#d24d3e' if gi==0 else '#2478b7'),s=20,alpha=.5)
   if vals:ax.plot([x-.25,x+.25],[np.mean(vals)]*2,c='black',lw=2)
 ax.set_title(ds);ax.set_xticks([0,1,3,4],['Qwen F','Qwen H','Gemma F','Gemma H']);ax.axhline(0,c='black',lw=.7);ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Mean thinking entropy difference (nats)');fig.suptitle('Fixed first64token exclusion sensitivity | user-wrong condition; F=flip, H=hold')
fig.savefig(W/'fixed64_sensitivity.png',dpi=160);plt.close(fig)
for example in P['examples']:
 model,key=example['model'],example['key'];rs=example['rows'];x=np.arange(1,len(rs)+1);fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained')
 for f,color,label in [('H_pressure','#d24d3e','With claim'),('H_neutral','#2478b7','Without claim, SAME history')]:
  y=np.array([float(r[f]) for r in rs]);axes[0].plot(x,y,c=color,lw=.6,alpha=.22);axes[0].plot(x[15:],np.convolve(y,np.ones(16)/16,'valid'),c=color,label=label,lw=1.4)
 for ax,f,color in [(axes[1],'delta_H','#7849a1'),(axes[2],'JSD','#188676')]:
  y=np.array([float(r[f]) for r in rs]);ax.plot(x,y,c=color,lw=.6,alpha=.25);ax.plot(x[15:],np.convolve(y,np.ones(16)/16,'valid'),c=color,lw=1.4);ax.set_ylabel(f+' (nats)')
 axes[0].set_ylabel('Entropy (nats)');axes[0].legend(fontsize=9);axes[1].axhline(0,c='black',lw=.6)
 for ax in axes:
  ax.axvspan(1,64,color='#adb7c2',alpha=.14);cut=example['opening_cutoff'];ax.axvline(cut+.5,c='#c08b2e',ls='--',lw=.9);ax.grid(alpha=.15)
 axes[-1].set_xlabel('Actual output token (1-based); orange dashed line = conservative opening cutoff');fig.suptitle(model+' | '+key)
 fig.savefig(W/(model+'__'+key+'.png'),dpi=130);plt.close(fig)
print('Rendered2 aggregate figures plus',len(P['examples']),'example trajectories,CPU only')
