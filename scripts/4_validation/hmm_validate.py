"""
hmm_validate.py - validation of the hidden Markov model

1. restarts from random initial values - does it converge to the same solution
2. model comparison - 1 state vs 2 states vs 3 states (BIC)
3. parameter recovery - are the true values recovered from simulated data
4. state separation - are the two states actually distinguishable
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import numpy as np, glob, csv, json, math, warnings
from scipy import stats
warnings.filterwarnings("ignore")
rng_global = np.random.default_rng(0)

RAW=str(RAW)
BEH={}
for p in glob.glob(f'{RAW}/**/*_behav.csv', recursive=True):
    pid=p.split('_')[-2]
    if pid not in BEH: BEH[pid]=list(csv.DictReader(open(p,encoding='utf-8-sig')))
pids=sorted(BEH)

def fit_hmm(obs, K=2, n_iter=120, seed=0, tol=1e-7):
    """K-state HMM, binary observations. Returns log-likelihood, transitions, emissions, posteriors"""
    rng=np.random.default_rng(seed); T=len(obs)
    A=rng.dirichlet(np.ones(K)*5, K)
    B=np.sort(rng.uniform(.1,.9,(K,1)),axis=0)
    B=np.hstack([1-B, B])                       # [P(0), P(1)]
    pi=rng.dirichlet(np.ones(K))
    ll_old=-np.inf
    for it in range(n_iter):
        al=np.zeros((T,K)); be=np.zeros((T,K)); sc=np.zeros(T)
        al[0]=pi*B[:,obs[0]]; sc[0]=al[0].sum()
        if sc[0]<=0: return None
        al[0]/=sc[0]
        for t in range(1,T):
            al[t]=(al[t-1]@A)*B[:,obs[t]]; sc[t]=al[t].sum()
            if sc[t]<=0: return None
            al[t]/=sc[t]
        be[-1]=1
        for t in range(T-2,-1,-1):
            be[t]=(A@(B[:,obs[t+1]]*be[t+1]))/sc[t+1]
        g=al*be; g/=g.sum(1,keepdims=True)
        xi=np.zeros((K,K))
        for t in range(T-1):
            m=(al[t][:,None]*A)*(B[:,obs[t+1]]*be[t+1])[None,:]
            s=m.sum()
            if s>0: xi+=m/s
        A=xi/np.maximum(xi.sum(1,keepdims=True),1e-12)
        for s_ in range(K):
            for o in range(2):
                B[s_,o]=g[obs==o,s_].sum()/max(g[:,s_].sum(),1e-12)
        B=np.clip(B,1e-4,1-1e-4); B/=B.sum(1,keepdims=True)
        pi=g[0]
        ll=np.log(sc).sum()
        if abs(ll-ll_old)<tol: break
        ll_old=ll
    return dict(ll=float(ll), A=A, B=B, g=g, pi=pi, iters=it+1)

def rep_seq(pid):
    ch=[r.get('choice') for r in BEH[pid]]
    return np.array([1 if (i>0 and ch[i]==ch[i-1]) else 0
                     for i in range(len(ch))])

# ---------- 1. random restarts ----------
print("="*76); print(" 1. 20 restarts from random initial values (K=2)"); print("="*76)
print(f"{'ID':>5}{'best logL':>12}{'same soln':>11}{'stick frac':>12}{'range':>16}{'separation':>12}")
BEST={}
for pid in pids:
    obs=rep_seq(pid)
    if len(obs)<40: continue
    runs=[]
    for s in range(20):
        r=fit_hmm(obs,2,seed=s)
        if r: runs.append(r)
    if not runs: continue
    lls=np.array([r['ll'] for r in runs])
    best=runs[int(np.argmax(lls))]
    same=int((np.abs(lls-lls.max())<0.5).sum())
    st=int(np.argmax(best['B'][:,1]))                # the state with the higher repeat probability
    fr=[float((r['g'][:,int(np.argmax(r['B'][:,1]))]>.5).mean()) for r in runs]
    sep=float(abs(best['B'][0,1]-best['B'][1,1]))    # difference in repeat probability between the states
    BEST[pid]=dict(ll=best['ll'], A=best['A'].tolist(), B=best['B'].tolist(),
                   post=best['g'][:,st].tolist(), frac=float((best['g'][:,st]>.5).mean()),
                   sep=sep, same=same)
    print(f"{pid:>5}{best['ll']:>12.2f}{same:>7}/20{(best['g'][:,st]>.5).mean()*100:>9.1f}%"
          f"{f'{min(fr)*100:.0f}–{max(fr)*100:.0f}%':>16}{sep:>9.2f}")

# ---------- 2. model comparison ----------
print("\n"+"="*76); print(" 2. comparison of the number of states (BIC, lower is better)"); print("="*76)
print(f"{'ID':>5}{'K=1':>11}{'K=2':>11}{'K=3':>11}{'chosen':>9}{'dBIC':>10}")
CHOICE={}
for pid in BEST:
    obs=rep_seq(pid); T=len(obs); bic={}
    # K=1: Bernoulli
    p1=obs.mean(); p1=min(max(p1,1e-6),1-1e-6)
    ll1=(obs*np.log(p1)+(1-obs)*np.log(1-p1)).sum()
    bic[1]=-2*ll1+1*np.log(T)
    for K in [2,3]:
        rr=[fit_hmm(obs,K,seed=s) for s in range(6)]
        rr=[r for r in rr if r]
        if not rr: continue
        ll=max(r['ll'] for r in rr)
        npar=K*(K-1)+K+(K-1)                      # transitions + emissions + initial
        bic[K]=-2*ll+npar*np.log(T)
    k=min(bic,key=bic.get)
    others=[v for kk,v in bic.items() if kk!=k]
    CHOICE[pid]=k
    print(f"{pid:>5}{bic.get(1,float('nan')):>11.1f}{bic.get(2,float('nan')):>11.1f}"
          f"{bic.get(3,float('nan')):>11.1f}{k:>7}{min(others)-bic[k]:>10.1f}")
from collections import Counter
print(f"\n  distribution of the model chosen: {dict(Counter(CHOICE.values()))}")

# ---------- 3. parameter recovery ----------
print("\n"+"="*76); print(" 3. parameter recovery (200 simulated sets, 120 trials)"); print("="*76)
def simulate(A,B,T,rng):
    s=0 if rng.random()<.5 else 1; obs=[]; states=[]
    for t in range(T):
        obs.append(int(rng.random()<B[s,1])); states.append(s)
        s=int(rng.random()>=A[s,0]) if s==0 else int(rng.random()<A[1,1])
    return np.array(obs), np.array(states)
rng=np.random.default_rng(1)
tru=[];est=[];cor=[]
for i in range(80):
    a00=rng.uniform(.7,.98); a11=rng.uniform(.7,.98)
    A=np.array([[a00,1-a00],[1-a11,a11]])
    b0=rng.uniform(.2,.5); b1=rng.uniform(.7,.98)
    B=np.array([[1-b0,b0],[1-b1,b1]])
    obs,states=simulate(A,B,120,rng)
    true_frac=float((states==1).mean())
    rr=[fit_hmm(obs,2,seed=s) for s in range(4)]
    rr=[r for r in rr if r]
    if not rr: continue
    best=rr[int(np.argmax([r['ll'] for r in rr]))]
    st=int(np.argmax(best['B'][:,1]))
    est_frac=float((best['g'][:,st]>.5).mean())
    tru.append(true_frac); est.append(est_frac)
    cor.append(float((( best['g'][:,st]>.5).astype(int)==states).mean()))
r=stats.spearmanr(tru,est)
print(f"  true stick fraction x estimated stick fraction   rho = {r.statistic:+.3f}  (p={r.pvalue:.4g}, n={len(tru)})")
print(f"  Pearson r = {stats.pearsonr(tru,est)[0]:+.3f}")
print(f"  mean absolute error {np.mean(np.abs(np.array(tru)-np.array(est)))*100:.1f} percentage points")
print(f"  median per-trial state classification accuracy {np.median(cor)*100:.1f}%")

# ---------- 4. diagnosis of the 0% participants ----------
print("\n"+"="*76); print(" 4. diagnosis of participants at 0% perseveration"); print("="*76)
print(f"{'ID':>5}{'repeat frac':>13}{'longest run':>13}{'state1 P(rep)':>15}{'state2 P(rep)':>15}{'separation':>12}")
for pid in BEST:
    obs=rep_seq(pid)
    runs=0; mx=0
    for o in obs:
        runs = runs+1 if o==1 else 0
        mx=max(mx,runs)
    B=np.array(BEST[pid]['B'])
    print(f"{pid:>5}{obs.mean()*100:>9.1f}%{mx:>10}"
          f"{B[0,1]:>12.2f}{B[1,1]:>12.2f}{BEST[pid]['sep']:>9.2f}")

json.dump(dict(best={k:{kk:vv for kk,vv in v.items()} for k,v in BEST.items()},
               choice=CHOICE, recov=dict(true=tru,est=est,acc=cor)),
          open(f'{DATA}/hmm_valid.json','w'))
print("\nwritten")
