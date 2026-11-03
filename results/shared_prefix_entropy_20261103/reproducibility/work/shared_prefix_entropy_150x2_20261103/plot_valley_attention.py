from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent
for e in json.loads((W/'two_cases_attention_alignment.json').read_text()):
 rs=e['rows'];x=[r['step'] for r in rs];fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
 for f,c,label in [('H_with_U','#d04c3c','With original claim'),('H_without_U','#2476b6','Without original claim, same history')]:axes[0].plot(x,[r[f] for r in rs],c=c,marker='.',lw=1,label=label)
 axes[0].set_ylabel('Next-token entropy (nats)');axes[0].legend(fontsize=9)
 axes[1].plot(x,[r['delta_H'] for r in rs],c='#7546a1',marker='.',lw=1.2);axes[1].axhline(0,c='#888',lw=.6);axes[1].axhline(-1,c='#888',ls=':',lw=.8);axes[1].set_ylabel('Entropy difference (nats)')
 for f,c,label in [('pressure_attention_percent','#e27e26','User-claim prompt span'),('image_attention_percent','#188773','Image tokens')]:axes[2].plot(x,[r[f] for r in rs],c=c,marker='.',lw=1.1,label=label)
 axes[2].set_ylabel('Mean attention mass (%)');axes[2].legend(fontsize=9)
 for ax in axes:ax.axvline(e['center'],c='black',ls='--',lw=.9);ax.grid(alpha=.18)
 axes[-1].set_xlabel('Predicted output token position (1-based); dashed = about to emit "user"')
 fig.suptitle(e['id']+' | '+('adopted wrong claim' if e['id']=='CQ-003' else 'rejected wrong claim')+' | local raw values, no smoothing')
 fig.savefig(W/(e['id']+'_valley_attention.png'),dpi=160);plt.close(fig)
print('Rendered2local valley/attention figures on CPU')
