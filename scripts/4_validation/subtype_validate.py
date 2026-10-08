"""
subtype_validate.py - validation of the clustering result
 1. participant resampling (the standard in the literature: random 80% subsets, adjusted Rand index)
 2. confound check - age, medication, registered disability type, signal quality
 3. continuous versus categorical - is the cluster structure actually discrete
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import json, numpy as np, warnings, math
from itertools import combinations
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
warnings.filterwarnings("ignore")

D=json.load(open(f'{DATA}/subtype.json'))
pids=D['pids']; Xz=np.array(D['Xz']); lab=np.array(D['lab2'])
FEAT=[f[1] for f in D['feat']]; n=len(pids)

# The demographic information (age, sex, disability type, medication) is participant data that
# the pipeline does not produce, so it is not in the repository. Section 2 (the confound check)
# runs only when data/demographics.csv is present and is skipped otherwise. The rest still runs.
import cohort
P=cohort.demographics(str(DATA)) or {}
HAVE_DEMO=bool(P)

print("="*76)
print(" 1. participant resampling (the standard method in the literature)")
print("="*76)
rng=np.random.default_rng(0)
def cl(X,k=2): return fcluster(linkage(pdist(X),'average'),k,criterion='maxclust')
base=cl(Xz)
ARI=[]; AMI=[]
B=500; keep=int(round(n*0.8))
for _ in range(B):
    idx=rng.choice(n,keep,replace=False)
    sub=cl(Xz[idx])
    ARI.append(adjusted_rand_score(base[idx],sub))
    AMI.append(adjusted_mutual_info_score(base[idx],sub))
ARI=np.array(ARI); AMI=np.array(AMI)
print(f"  {keep}/{n} participants per subset - {B} resamples")
print(f"  adjusted Rand index      median {np.median(ARI):+.3f}  "
      f"IQR [{np.percentile(ARI,25):+.3f}, {np.percentile(ARI,75):+.3f}]")
print(f"  adjusted mutual information  median {np.median(AMI):+.3f}  "
      f"IQR [{np.percentile(AMI,25):+.3f}, {np.percentile(AMI,75):+.3f}]")
print(f"  exact agreement (ARI=1) {np.mean(ARI>0.99)*100:.0f}%")
print(f"  chance level (ARI<0.1)  {np.mean(ARI<0.1)*100:.0f}%")

# with 004 and 005 excluded
for drop in [['004'],['005'],['004','005']]:
    ix=[i for i,p in enumerate(pids) if p not in drop]
    sub=cl(Xz[ix]); sizes=sorted(np.bincount(sub)[1:], reverse=True)
    who=' | '.join('{'+', '.join(pids[ix[j]] for j in range(len(ix)) if sub[j]==c)+'}'
                   for c in np.unique(sub))
    print(f"\n  with {drop} excluded, k=2: sizes {sizes}  {who}")

print("\n"+"="*76)
print(" 2. confound check")
print("="*76)
if not HAVE_DEMO:
    print("  Skipped - age, sex, disability type and medication are participant data and are not in the repository.")
    print(f"  To run it, put the following columns in data/{cohort.DEMO_FILE}:")
    print("    id, age, sex, disability, psychotropic_med")
# No script in this repository produces trialwise2.json.
# Without it only the trial-yield column is left empty; the rest of the validation still runs.
try:
    Q=json.load(open(f'{DATA}/trialwise2.json'))
    YIELD={p: sum(1 for x in Q[p] if x)/len(Q[p]) for p in Q}
except FileNotFoundError:
    YIELD={}
    print("  Note. trialwise2.json is absent, so the trial-yield column is left empty.")
def num(v):
    try: return float(v)
    except (TypeError,ValueError): return None
def _yld_s(v):
    return '—' if v is None else format(v * 100, '.0f') + '%'

# Without the demographic information those columns must not be filled with 0 and reported.
# A missing value is not 0, and doing so manufactures a result that does not exist, such as
# "autism registration 0.00 vs 0.00 (p = 1.000)". What follows runs only when the data are there.
# The disability column is free text. A value counts when it contains one of these substrings,
# matched case-insensitively. Set ASD_DISABILITY_AUTISM / ASD_DISABILITY_DOWN (comma separated)
# when the column is written in another language.
AUTISM_WORDS=[s.strip().lower() for s in os.environ.get('ASD_DISABILITY_AUTISM','autis').split(',') if s.strip()]
DOWN_WORDS=[s.strip().lower() for s in os.environ.get('ASD_DISABILITY_DOWN','down,trisomy').split(',') if s.strip()]
KEYS=[('age','age'),('med','medication'),('autism','autism registration'),
      ('down','Down syndrome'),('yld','trial yield')]
if HAVE_DEMO:
    rows=[]
    for i,p in enumerate(pids):
        d=P.get(p,{})
        dis=str(d.get('disability') or '')
        rows.append(dict(pid=p, grp=int(lab[i]), age=num(d.get('age')),
                         sex=d.get('sex'), med=num(d.get('psychotropic_med')),
                         autism=1 if any(w in dis.lower() for w in AUTISM_WORDS) else 0,
                         down=1 if any(w in dis.lower() for w in DOWN_WORDS) else 0,
                         yld=YIELD.get(p)))
    print(f"{'ID':>5}{'cluster':>9}{'age':>6}{'sex':>5}{'med':>5}{'autism':>8}"
          f"{'Down':>6}{'yield':>8}")
    for r in sorted(rows,key=lambda z:(z['grp'],z['pid'])):
        _a='—' if r['age'] is None else format(r['age'],'.0f')
        _m='—' if r['med'] is None else format(r['med'],'.0f')
        print(f"{r['pid']:>5}{r['grp']:>5}{_a:>6}{str(r['sex'] or '—'):>5}"
              f"{_m:>5}{r['autism']:>9}{r['down']:>6}{_yld_s(r['yld']):>9}")
else:
    # The trial yield has nothing to do with the demographics, so it is reported when present.
    rows=[dict(pid=p, grp=int(lab[i]), yld=YIELD.get(p))
          for i,p in enumerate(pids)]
    KEYS=[('yld','trial yield')]
    if YIELD:
        print(f"{'ID':>5}{'cluster':>9}{'yield':>8}")
        for r in sorted(rows,key=lambda z:(z['grp'],z['pid'])):
            print(f"{r['pid']:>5}{r['grp']:>5}{_yld_s(r['yld']):>9}")

g1=[r for r in rows if r['grp']==1]; g2=[r for r in rows if r['grp']==2]
_done=False
for key,nm in KEYS:
    a=[r[key] for r in g1 if r.get(key) is not None]
    b=[r[key] for r in g2 if r.get(key) is not None]
    if not a or not b: continue
    if not _done:
        print(f"\n{'variable':>22}{'cluster 1':>12}{'cluster 2':>12}{'test':>18}"); _done=True
    if key in ('med','autism','down'):
        try: p_=stats.fisher_exact([[sum(a),len(a)-sum(a)],[sum(b),len(b)-sum(b)]])[1]
        except Exception: p_=float('nan')
        print(f"{nm:>12}{sum(a)/len(a):>12.2f}{sum(b)/len(b):>12.2f}"
              f"{'Fisher p='+f'{p_:.3f}':>18}")
    else:
        u=stats.mannwhitneyu(a,b).pvalue
        print(f"{nm:>12}{np.mean(a):>12.2f}{np.mean(b):>12.2f}"
              f"{'MWU p='+f'{u:.3f}':>18}")
if not _done:
    print("  no variable available to compare.")

print("\n"+"="*76)
print(" 3. continuous versus categorical - is the structure discrete")
print("="*76)
# Gap-statistic-like: compare within-cluster variance against a uniform reference
def wss(X,k):
    l=cl(X,k) if k>1 else np.ones(len(X),int)
    return sum(((X[l==c]-X[l==c].mean(0))**2).sum() for c in np.unique(l))
obs=[wss(Xz,k) for k in range(1,5)]
ref=np.zeros((200,4))
lo,hi=Xz.min(0),Xz.max(0)
for b in range(200):
    R=rng.uniform(lo,hi,Xz.shape)
    for k in range(1,5): ref[b,k-1]=wss(R,k)
gap=np.log(ref).mean(0)-np.log(obs)
sk=np.log(ref).std(0)*math.sqrt(1+1/200)
print(f"{'k':>3}{'observed WSS':>14}{'gap':>9}{'SE':>10}")
for k in range(1,5):
    print(f"{k:>3}{obs[k-1]:>12.1f}{gap[k-1]:>+9.3f}{sk[k-1]:>10.3f}")
best=[k for k in range(1,4) if gap[k-1]>=gap[k]-sk[k]]
print(f"  optimal k by the gap criterion = {best[0] if best else 'undetermined'}"
      "   (1 means there is no cluster structure)")

# Bimodality check - the first principal component
from numpy.linalg import svd
U,s,Vt=svd(Xz-Xz.mean(0),full_matrices=False)
pc1=U[:,0]*s[0]
dip=stats.kurtosis(pc1)
print(f"\n  kurtosis of PC1 {dip:+.2f}  (negative = bimodal or flat, positive = unimodal with tails)")
print("  PC1 values: " + " ".join(f"{p}:{v:+.2f}" for p,v in zip(pids,pc1)))
print(f"  Shapiro-Wilk p = {stats.shapiro(pc1).pvalue:.3f} "
      "(small = departs from normality)")

json.dump(dict(ari=ARI.tolist(),ami=AMI.tolist(),
               gap=gap.tolist(),sk=sk.tolist(),pc1=pc1.tolist(),
               conf=[{k:(v if not isinstance(v,(np.integer,np.floating)) else float(v))
                      for k,v in r.items()} for r in rows]),
          open(f'{DATA}/subtype_valid.json','w'))
