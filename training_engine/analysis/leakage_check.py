"""How much the duplicate-exam leak inflated AUC on the old folds.

Each leaked exam's score with its copy in training (_preview_2026-11-04_old_folds) is compared
with its own or its identical twin's score in the fixed runs. Random negatives give the noise floor.

Usage (from training_engine/): python analysis/leakage_check.py
"""
import re, numpy as np, pandas as pd
from pathlib import Path
eid=lambda s: re.search(r'(BP\d{4}[A-Z]?)',s).group(1)
OLD=Path('experiments/_preview_2026-11-04_old_folds'); runs=sorted(p.name for p in OLD.iterdir())
def rd(p): return pd.read_csv(p).assign(e=lambda x:x.image.map(eid))
LEAK={'BP0250A':'BP0250A','BP0250B':'BP0250A','BP0959':'BP0959','BP0960':'BP0959','BP0423':'BP0423','BP0425':'BP0423','BP0734':'BP0734','BP0930':'BP0930'}
EXCL=set(LEAK)|{'BP0112','BP0361','BP0074','BP0464'}
def frac_pos_above(v):
    pos=np.sort(v[v.true_label==1].prob.to_numpy())
    return dict(zip(v.e,1-np.searchsorted(pos,v.prob.to_numpy(),'left')/len(pos)))
rows=[]
for r in runs:
    o={};n={};nneg={}
    for k in range(1,6):
        v=rd(OLD/r/f'fold_{k}/predictions.csv'); o.update(frac_pos_above(v)); nn=int((v.true_label==0).sum()); nneg.update({e:nn for e in v.e})
        w=rd(Path('experiments')/r/f'fold_{k}/predictions.csv'); n.update(frac_pos_above(w))
        if k==1: lab={}
        lab.update(dict(zip(v.e,v.true_label)))
    for e,l in lab.items():
        tw=LEAK.get(e,e)
        if l==0 and tw in n: rows.append((r,e,e in LEAK,(o[e]-n[tw])/nneg[e],o[e],n[tw]))
d=pd.DataFrame(rows,columns='run e leaked infl old new'.split())
print('LEAKED VAL NEGATIVES, fraction of fold positives scoring above it (higher = model finds it easier to call negative)')
print(d[d.leaked].groupby('e')[['old','new']].mean().round(3).assign(diff=lambda x:(x.old-x.new).round(3)).to_string())
per=d[d.leaked].groupby('run').infl.sum()/5
print('inflation of mean val AUC per run:',per.round(4).to_dict()); print('mean',round(per.mean(),4))
ctl=d[~d.leaked&~d.e.isin(EXCL)].pivot(index='e',columns='run',values='infl').dropna()
rng=np.random.default_rng(0); arr=ctl.to_numpy(); sims=np.array([arr[rng.choice(len(arr),8,replace=False)].sum(0).mean()/5 for _ in range(20000)])
print('control negatives',len(arr),'null mean',round(sims.mean(),4),'sd',round(sims.std(),4),'95% range',np.percentile(sims,[2.5,97.5]).round(4),'P(null>=obs)',round((sims>=per.mean()).mean(),3))
print('\nTEST BP0702 (positive): fraction of test negatives scoring below it, mean over 5 fold models')
T=[]
for r in runs:
    def fr(root):
        out={}
        for k in range(1,6):
            t=rd(root/r/f'fold_{k}/test_predictions.csv'); neg=np.sort(t[t.true_label==0].prob.to_numpy())
            for e,p,l in zip(t.e,t.prob,t.true_label):
                if l==1: out.setdefault(e,[]).append(np.searchsorted(neg,p,'left')/len(neg))
        return {e:np.mean(v) for e,v in out.items()}
    a,b=fr(OLD),fr(Path('experiments'))
    T.append(pd.Series({e:a[e]-b[e] for e in a},name=r)); print(f'  {r:15s} old {a["BP0702"]:.3f} new {b["BP0702"]:.3f}')
T=pd.DataFrame(T); m=T.mean()
print('BP0702 mean shift',round(m['BP0702'],3),'-> AUC inflation',round(m['BP0702']/23,4),'| other 22 positives: mean',round(m.drop('BP0702').mean(),3),'sd',round(m.drop('BP0702').std(),3),'rank of BP0702 shift (1=largest)',int((m>m['BP0702']).sum()+1),'of',len(m))
