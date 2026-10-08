"""
rlmodel.py - the preregistered reinforcement-learning model

Model (as registered in the plan)
  Q update:  on reward     Q_c += eta_rew   * (1 - Q_c)
             on no reward  Q_c += eta_unrew * (0 - Q_c)
  choice:    P(a) proportional to exp(beta*Q_a + kappa*[a == previous choice])

Parameters
  eta_rew    learning rate after reward
  eta_unrew  learning rate after no reward   <- sensitivity to negative feedback
  beta       inverse temperature
  kappa      stickiness                     <- a co-primary index

Order of validation (done first, so as not to repeat the HMM failure)
  1. parameter recovery - are the true values recovered from simulated data
  2. identifiability - can kappa and eta_unrew be told apart
  3. dependence on the starting point - 20 random starts
  4. model comparison - single learning rate / asymmetric / +kappa
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import numpy as np, glob, csv, json, math, warnings
from scipy.optimize import minimize
from scipy import stats
warnings.filterwarnings("ignore")

RAW=str(RAW)
BEH={}
for p in glob.glob(f'{RAW}/**/*_behav.csv', recursive=True):
    pid=p.split('_')[-2]
    if pid not in BEH: BEH[pid]=list(csv.DictReader(open(p,encoding='utf-8-sig')))
pids=sorted(BEH)

def load(pid):
    """Arrays of choice (0/1) and reward (0/1)"""
    B=BEH[pid]; ch=[]; rw=[]
    for r in B:
        c=r.get('choice')
        if c in (None,''): ch.append(None); rw.append(None); continue
        ch.append(c)
        try: rw.append(int(r.get('reward', r.get('correct',0))))
        except (TypeError,ValueError): rw.append(0)
    opts=sorted({c for c in ch if c is not None})
    if len(opts)!=2: return None,None
    a=np.array([opts.index(c) if c is not None else -1 for c in ch])
    r=np.array([x if x is not None else 0 for x in rw])
    m=a>=0
    return a[m], r[m]

def nll(par, a, r, model='full'):
    if model=='single':
        er=eu=1/(1+np.exp(-par[0])); b=np.exp(par[1]); k=0.
    elif model=='asym':
        er=1/(1+np.exp(-par[0])); eu=1/(1+np.exp(-par[1])); b=np.exp(par[2]); k=0.
    else:
        er=1/(1+np.exp(-par[0])); eu=1/(1+np.exp(-par[1]))
        b=np.exp(par[2]); k=par[3]
    Q=np.array([.5,.5]); L=0.; prev=-1
    for t in range(len(a)):
        v=b*Q.copy()
        if prev>=0: v[prev]+=k
        v-=v.max(); p=np.exp(v); p/=p.sum()
        L-=np.log(max(p[a[t]],1e-12))
        c=a[t]
        Q[c]+= (er*(1-Q[c])) if r[t]==1 else (eu*(0-Q[c]))
        prev=c
    return L

NP={'single':2,'asym':3,'full':4}
def fit(a, r, model='full', n_start=20, seed=0):
    rng=np.random.default_rng(seed); best=None
    for s in range(n_start):
        x0=np.r_[rng.normal(0,1.2,NP[model]-1), rng.normal(0,1.)][:NP[model]]
        try:
            res=minimize(nll,x0,args=(a,r,model),method='Nelder-Mead',
                         options=dict(maxiter=1500,xatol=1e-4,fatol=1e-4))
        except Exception: continue
        if res.success or res.fun<np.inf:
            if best is None or res.fun<best.fun: best=res
    if best is None: return None
    p=best.x
    out=dict(nll=float(best.fun), n=len(a))
    if model=='single':
        out.update(er=1/(1+np.exp(-p[0])), eu=1/(1+np.exp(-p[0])),
                   beta=float(np.exp(p[1])), kappa=0.)
    elif model=='asym':
        out.update(er=1/(1+np.exp(-p[0])), eu=1/(1+np.exp(-p[1])),
                   beta=float(np.exp(p[2])), kappa=0.)
    else:
        out.update(er=1/(1+np.exp(-p[0])), eu=1/(1+np.exp(-p[1])),
                   beta=float(np.exp(p[2])), kappa=float(p[3]))
    out['bic']=2*out['nll']+NP[model]*math.log(len(a))
    return out

def simulate(er,eu,b,k,T=120,seg=(19,17,23,17,22),seed=0):
    rng=np.random.default_rng(seed)
    corr=0; bounds=np.cumsum(seg); Q=np.array([.5,.5]); prev=-1
    A=[];R=[]
    for t in range(T):
        for i,bd in enumerate(bounds):
            if t<bd: corr=i%2; break
        v=b*Q.copy()
        if prev>=0: v[prev]+=k
        v-=v.max(); p=np.exp(v); p/=p.sum()
        c=int(rng.random()<p[1])
        rw=int(rng.random()<(.8 if c==corr else .2))
        A.append(c); R.append(rw)
        Q[c]+= (er*(1-Q[c])) if rw==1 else (eu*(0-Q[c]))
        prev=c
    return np.array(A), np.array(R)

# ---------- 1. parameter recovery ----------
print("="*80); print(" 1. parameter recovery (150 simulated sets)"); print("="*80)
rng=np.random.default_rng(1); T={'er':[],'eu':[],'beta':[],'kappa':[]}
E={'er':[],'eu':[],'beta':[],'kappa':[]}
for i in range(60):
    er=rng.uniform(.05,.8); eu=rng.uniform(.05,.8)
    b=rng.uniform(1.,8.); k=rng.uniform(-.5,3.)
    a,r=simulate(er,eu,b,k,seed=1000+i)
    f=fit(a,r,'full',n_start=4,seed=i)
    if f is None: continue
    for key,tv in [('er',er),('eu',eu),('beta',b),('kappa',k)]:
        T[key].append(tv); E[key].append(f[key])
print(f"{'parameter':>12}{'true x estimated':>20}{'Pearson r':>12}{'mean abs error':>16}")
for key,lab in [('er','eta_rew'),('eu','eta_unrew'),('beta','beta'),('kappa','kappa')]:
    t=np.array(T[key]); e=np.array(E[key])
    rs=stats.spearmanr(t,e).statistic; rp=stats.pearsonr(t,e)[0]
    print(f"{lab:>10}{rs:>+14.3f}{rp:>12.3f}{np.mean(np.abs(t-e)):>14.3f}")

# ---------- 2. identifiability ----------
print("\n"+"="*80); print(" 2. identifiability - correlation between the estimates"); print("="*80)
K=['er','eu','beta','kappa']; L=['eta_rew','eta_unrew','beta','kappa']
Em=np.array([E[k] for k in K])
print("        " + "".join(f"{l:>10}" for l in L))
for i,l in enumerate(L):
    print(f"{l:>7} " + "".join(f"{np.corrcoef(Em[i],Em[j])[0,1]:>10.2f}" for j in range(4)))
print("\n  off-diagonal |r| > .6 means the two parameters are hard to tell apart")

# ---------- 3-4. real data ----------
print("\n"+"="*80); print(" 3. fit to the real data and model comparison"); print("="*80)
FIT={}
print(f"{'ID':>5}{'trials':>8}{'eta_rew':>10}{'eta_unrew':>11}{'beta':>8}{'kappa':>8}"
      f"{'BIC single':>12}{'BIC asym':>11}{'BIC full':>10}{'chosen':>9}")
for pid in pids:
    a,r=load(pid)
    if a is None or len(a)<50: continue
    fs={m:fit(a,r,m,n_start=10,seed=7) for m in ['single','asym','full']}
    if any(v is None for v in fs.values()): continue
    best=min(fs,key=lambda m: fs[m]['bic'])
    f=fs['full']; FIT[pid]=dict(full=f, bic={m:fs[m]['bic'] for m in fs}, best=best)
    print(f"{pid:>5}{len(a):>5}{f['er']:>9.3f}{f['eu']:>10.3f}{f['beta']:>8.2f}"
          f"{f['kappa']:>8.2f}{fs['single']['bic']:>10.1f}{fs['asym']['bic']:>11.1f}"
          f"{fs['full']['bic']:>10.1f}{best:>7}")
from collections import Counter
print(f"\n  distribution of the model chosen: {dict(Counter(v['best'] for v in FIT.values()))}")

# dependence on the starting point
print("\n"+"="*80); print(" 4. dependence on the starting point (times the optimum was reached in 10 random starts)"); print("="*80)
print(f"{'ID':>5}{'best nll':>11}{'matches':>9}{'kappa range':>20}")
for pid in list(FIT)[:6]:
    a,r=load(pid); vals=[]
    for s in range(10):
        f=fit(a,r,'full',n_start=1,seed=s)
        if f: vals.append((f['nll'],f['kappa']))
    if not vals: continue
    mn=min(v[0] for v in vals)
    same=sum(1 for v in vals if v[0]-mn<0.5)
    ks=[v[1] for v in vals if v[0]-mn<0.5]
    print(f"{pid:>5}{mn:>11.2f}{same:>6}/10"
          f"{f'{min(ks):+.2f} ~ {max(ks):+.2f}':>18}")

json.dump({p:{'full':FIT[p]['full'],'bic':FIT[p]['bic'],'best':FIT[p]['best']}
           for p in FIT}, open(f'{DATA}/rlfit.json','w'))
print("\nwritten")
