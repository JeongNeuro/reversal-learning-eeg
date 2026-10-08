"""
network_recompute.py - recompute the network indices from conn.json

Definitions
  - synchronisation  PLV (no bias correction)
  - bands            theta 4-8, alpha 8-12, beta 14-30 Hz (matched to the bin edges of conn.json)
  - path length      weighted graph, distance = 1/PLV, shortest paths by Floyd-Warshall
  - clustering       weighted (Onnela et al. 2005), geometric-mean triangles
  - node degree      number of edges above the top-40% threshold (the 60th percentile of all rest and task edge values)
  - test             Wilcoxon signed-rank (two-sided)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA

import json, numpy as np, warnings
from scipy import stats
warnings.filterwarnings("ignore")

CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
C=json.load(open(f'{DATA}/conn.json'))

SEED, NB = 1, 10000          # the repository-wide convention


def boot_mean(d, seed=SEED, nb=NB):
    """Percentile bootstrap interval of the mean paired difference. The value reported in manuscript 3.4."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(nb)], [2.5, 97.5])
pids=sorted(C)
BANDS={'theta':['4-6','6-8'],'alpha':['8-10','10-12'],
       'beta':['14-16','16-18','18-20','20-22','22-24','24-26','26-28','28-30']}

def mat(pid,cond,bands):
    ch=C[pid]['ch']; acc=[]
    for b in bands:
        m=C[pid][cond].get(b)
        if m: acc.append(np.array(m))
    if not acc: return None,None
    return np.mean(acc,axis=0), ch

def mean_plv(M):
    iu=np.triu_indices(M.shape[0],1)
    return float(np.nanmean(M[iu]))

def wpath(M):
    """Weighted characteristic path length, distance = 1/PLV"""
    n=M.shape[0]; D=np.where(M>0,1.0/np.maximum(M,1e-9),np.inf)
    np.fill_diagonal(D,0)
    for k in range(n):
        D=np.minimum(D,D[:,k][:,None]+D[k,:][None,:])
    iu=np.triu_indices(n,1); v=D[iu]; v=v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan

def wclust(M):
    """Weighted clustering coefficient (Onnela)"""
    n=M.shape[0]; W=M.copy(); np.fill_diagonal(W,0)
    mx=W.max() if W.max()>0 else 1
    A=(W>0).astype(float); K=A.sum(1)
    W13=(W/mx)**(1/3)
    cyc=np.diag(W13@W13@W13)
    with np.errstate(divide='ignore',invalid='ignore'):
        c=cyc/(K*(K-1))
    return float(np.nanmean(c))

def degree(M,thr):
    A=(M>thr).astype(int); np.fill_diagonal(A,0)
    return float(A.sum(1).mean())

print("="*78)
print(" recomputed from conn.json")
print("="*78)
print(f"{len(pids)} participants: {', '.join(pids)}\n")

RES={}
for band,bs in BANDS.items():
    rows=[]
    for p in pids:
        Mr,_=mat(p,'rest',bs); Mt,_=mat(p,'task',bs)
        if Mr is None or Mt is None: continue
        allv=np.concatenate([Mr[np.triu_indices(len(Mr),1)],
                             Mt[np.triu_indices(len(Mt),1)]])
        thr=np.nanpercentile(allv,60)
        rows.append(dict(pid=p,
            plv_r=mean_plv(Mr), plv_t=mean_plv(Mt),
            pl_r=wpath(Mr), pl_t=wpath(Mt),
            cc_r=wclust(Mr), cc_t=wclust(Mt),
            dg_r=degree(Mr,thr), dg_t=degree(Mt,thr)))
    RES[band]=rows
    print(f"[{band}]  n = {len(rows)}")
    print(f"{'index':>22}{'rest':>9}{'task':>9}{'change':>9}"
          f"{'95% CI':>22}{'same dir.':>11}{'Wilcoxon p':>13}")
    for key,lab in [('plv','mean PLV'),('pl','path length (weighted)'),
                    ('cc','clustering (weighted)'),('dg','node degree')]:
        a=np.array([r[f'{key}_r'] for r in rows]); b=np.array([r[f'{key}_t'] for r in rows])
        d=b-a
        w=stats.wilcoxon(d).pvalue
        same=max(sum(1 for x in d if x>0),sum(1 for x in d if x<0))
        lo,hi=boot_mean(d)
        print(f"{lab:>14}{a.mean():>9.3f}{b.mean():>9.3f}{d.mean():>+9.3f}"
              f"{'[%+.3f, %+.3f]'%(lo,hi):>22}{same:>7}/{len(d)}{w:>13.4f}")
    print()

print("="*78)
print(" smallest attainable Wilcoxon p (two-sided)")
print("="*78)
for n in [9,10,11,12]:
    print(f"  n = {n}: smallest p = {2/2**n:.5f}")

print("\n"+"="*78)
print(" mean alpha PLV under each band definition")
print("="*78)
for lab,bs in [('8–10 Hz',['8-10']),('10–12 Hz',['10-12']),
               ('8–12 Hz',['8-10','10-12']),('8–14 Hz',['8-10','10-12','12-14'])]:
    a=[];b=[]
    for p in pids:
        Mr,_=mat(p,'rest',bs); Mt,_=mat(p,'task',bs)
        if Mr is None: continue
        a.append(mean_plv(Mr)); b.append(mean_plv(Mt))
    print(f"  {lab:>12}: rest {np.mean(a):.3f} -> task {np.mean(b):.3f}")

print("\n"+"="*78)
print(" if the graph had been binary (for reference)")
print("="*78)
for p in pids[:1]:
    Mr,_=mat(p,'rest',BANDS['alpha'])
    n=len(Mr)                                   # can be below 14 once bad channels are excluded
    allv=Mr[np.triu_indices(n,1)]
    thr=np.nanpercentile(allv,60)
    A=(Mr>thr).astype(int); np.fill_diagonal(A,0)
    D=np.where(A>0,1.0,np.inf); np.fill_diagonal(D,0)
    for k in range(n): D=np.minimum(D,D[:,k][:,None]+D[k,:][None,:])
    iu=np.triu_indices(n,1); v=D[iu]; v=v[np.isfinite(v)]
    print(f"  {p} alpha binary graph ({n} channels): mean degree {A.sum(1).mean():.1f} - "
          f"path length {v.mean():.2f}")
print("  -> in a binary graph with a degree of about 7, the path length can hardly exceed 1.5")

json.dump({b:RES[b] for b in RES}, open(f'{DATA}/network_recomputed.json','w'))
