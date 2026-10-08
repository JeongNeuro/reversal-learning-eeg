"""
trajectory.py - two directions

A. reversal-aligned learning trajectories (the approach of Wang et al.)
   - average correctness per participant over the trials around each reversal
   - cluster the curves themselves (their shape, not a summary score)

B. trial-level strategy estimation
   - hidden Markov model: two states, 'following feedback' and 'perseverating'
   - posterior probability of being in the perseverating state on each trial
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import numpy as np, glob, csv, json, math, warnings
from collections import Counter
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
warnings.filterwarnings("ignore")

RAW=str(RAW)
BEH={}
for p in glob.glob(f'{RAW}/**/*_behav.csv', recursive=True):
    pid=p.split('_')[-2]
    if pid in BEH: continue
    BEH[pid]=list(csv.DictReader(open(p, encoding='utf-8-sig')))
pids=sorted(BEH)
print(f"behavioural data for {len(pids)} participants\n")

def gi(r,k,d=None):
    try: return int(r[k])
    except (KeyError,ValueError,TypeError): return d

# ---------- A. reversal-aligned trajectories ----------
PRE, POST = 8, 20                    # 8 trials before a reversal, 20 after
off=np.arange(-PRE, POST)
CURVE={}
for pid in pids:
    B=BEH[pid]
    rev=[i for i,r in enumerate(B) if r.get('is_reversal_trial')=='1']
    if len(rev)<3: continue
    mat=[]
    for i in rev:
        seg=[]
        for o in off:
            j=i+o
            seg.append(gi(B[j],'correct') if 0<=j<len(B) else np.nan)
        mat.append(seg)
    C=np.nanmean(np.array(mat,float),axis=0)
    if np.isnan(C).sum()>3: continue
    # interpolate the gaps
    idx=np.arange(len(C)); m=~np.isnan(C)
    C=np.interp(idx,idx[m],C[m])
    CURVE[pid]=C
    print(f"  {pid}: {len(rev)} reversals - accuracy before reversal {C[:PRE].mean():.2f} "
          f"- first 5 after {C[PRE:PRE+5].mean():.2f} - late {C[-8:].mean():.2f}")

cp=sorted(CURVE); X=np.array([CURVE[p] for p in cp])
print(f"\ntrajectories obtained for {len(cp)} participants")

print("\n"+"="*74); print(" A. trajectory clustering"); print("="*74)
Z=linkage(pdist(X,'euclidean'),'ward')
from scipy.cluster.hierarchy import cophenet
coph,_=cophenet(Z,pdist(X,'euclidean'))
print(f"  cophenetic correlation {coph:.3f}")
for k in [2,3,4]:
    lab=fcluster(Z,k,criterion='maxclust')
    grp=' | '.join('{'+', '.join(p for p,l in zip(cp,lab) if l==c)+'}'
                   for c in range(1,k+1))
    print(f"  k={k}: {grp}")

# bootstrap (participant resampling)
rng=np.random.default_rng(0)
from sklearn.metrics import adjusted_rand_score
base=fcluster(Z,3,criterion='maxclust')
ari=[]
for _ in range(500):
    ix=rng.choice(len(cp),int(len(cp)*0.8),replace=False)
    sub=fcluster(linkage(pdist(X[ix]),'ward'),3,criterion='maxclust')
    ari.append(adjusted_rand_score(base[ix],sub))
print(f"\n  participant resampling 80%, 500 resamples, median adjusted Rand index {np.median(ari):+.3f}")
print(f"  IQR [{np.percentile(ari,25):+.3f}, {np.percentile(ari,75):+.3f}]")

lab3=fcluster(Z,3,criterion='maxclust')
print("\n  trajectory features by cluster")
print(f"{'cluster':>8}{'n':>4}{'before':>9}{'first 5':>9}{'late 8':>9}{'recovery':>10}  participants")
for c in range(1,4):
    ix=[i for i in range(len(cp)) if lab3[i]==c]
    if not ix: continue
    A=X[ix]
    pre=A[:,:PRE].mean(); imm=A[:,PRE:PRE+5].mean(); late=A[:,-8:].mean()
    print(f"{c:>5}{len(ix):>4}{pre:>9.2f}{imm:>9.2f}{late:>9.2f}{late-imm:>+9.2f}  "
          f"{', '.join(cp[i] for i in ix)}")

# ---------- B. trial-level strategy estimation (2-state HMM) ----------
print("\n"+"="*74); print(" B. trial-level strategy estimation"); print("="*74)
print("  state 1 = following feedback (switch after X, stay after O)")
print("  state 2 = perseverating (repeat the previous choice)\n")

def hmm(obs, n_iter=60):
    """obs: per trial, is this the same choice as the previous one, 0/1. Fits a 2-state HMM."""
    T=len(obs)
    A=np.array([[.9,.1],[.1,.9]])           # transitions
    B=np.array([[.4,.6],[.05,.95]])         # P(repeat | state)  row 1 = following, row 2 = perseverating
    pi=np.array([.5,.5])
    for _ in range(n_iter):
        # forward-backward
        al=np.zeros((T,2)); be=np.zeros((T,2)); sc=np.zeros(T)
        al[0]=pi*B[:,obs[0]]; sc[0]=al[0].sum(); al[0]/=sc[0]
        for t in range(1,T):
            al[t]=(al[t-1]@A)*B[:,obs[t]]; sc[t]=al[t].sum(); al[t]/=sc[t]
        be[-1]=1
        for t in range(T-2,-1,-1):
            be[t]=(A@(B[:,obs[t+1]]*be[t+1]))/sc[t+1]
        g=al*be; g/=g.sum(1,keepdims=True)
        xi=np.zeros((2,2))
        for t in range(T-1):
            m=(al[t][:,None]*A)*(B[:,obs[t+1]]*be[t+1])[None,:]
            xi+=m/m.sum()
        A=xi/xi.sum(1,keepdims=True)
        for s in range(2):
            for o in range(2):
                B[s,o]=g[obs==o,s].sum()/g[:,s].sum()
        B=np.clip(B,1e-4,1-1e-4); B/=B.sum(1,keepdims=True)
        pi=g[0]
    # the perseverating state is the one with the higher repeat probability
    st=int(np.argmax(B[:,1]))
    return g[:,st], A, B

STRAT={}
for pid in pids:
    B_=BEH[pid]
    ch=[r.get('choice') for r in B_]
    rep=np.array([1 if (i>0 and ch[i]==ch[i-1]) else 0 for i in range(len(ch))])
    if len(rep)<40: continue
    try:
        post,Am,Bm=hmm(rep)
    except Exception: continue
    STRAT[pid]=dict(post=post.tolist(),
                    frac=float((post>.5).mean()),
                    stay_A=float(Am[0,0]), stay_B=float(Am[1,1]))
    print(f"  {pid}: trials in the perseverating state {(post>.5).mean()*100:>5.1f}%  "
          f"- state persistence following {Am[0,0]:.2f} / perseverating {Am[1,1]:.2f}")

json.dump(dict(curves={p:CURVE[p].tolist() for p in cp}, off=off.tolist(),
               lab3=lab3.tolist(), cp=cp, ari=ari, strat=STRAT),
          open(f'{DATA}/trajectory.json','w'))

# -- relation of the two directions to the EEG --
# The sample and the behavioural indices come from cohort.py alone (the dependency on the
# workbook's analysis sheet is removed - that file is participant data and is not published)
import cohort
M=cohort.by_id(str(DATA))
print("\n"+"="*74); print(" proportion in the perseverating state and the EEG indices"); print("="*74)
for key,lab in [('front_exp','frontal slope'),('front_alpha','frontal alpha')]:
    x=[];y=[]
    for p,d in STRAT.items():
        v=M.get(p,{}).get(key)
        if v is not None and not M.get(p,{}).get('excl_winstay'):
            x.append(v); y.append(d['frac'])
    if len(x)>=6:
        r=stats.spearmanr(x,y)
        print(f"  {lab:<16} x perseverating proportion   rho = {r.statistic:+.3f}  "
              f"p = {r.pvalue:.4f}  (n={len(x)})")
