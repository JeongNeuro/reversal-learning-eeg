"""
subtype.py - build participant profiles from the significant indices and type them

Principles for selecting the indices
  - drawn from results that were significant
  - axes that do not duplicate each other (reflecting the double dissociation found earlier)
  - few missing values

Validation
  - how often participants fall in the same cluster under bootstrap resampling
  - whether the clusters are stable in this sample is itself treated as the result
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import json, numpy as np, warnings, math
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")
from fooof import FOOOF

# The sample and the behavioural indices come from cohort.py alone (the dependency on the
# workbook's analysis sheet is removed - that file is participant data and is not published)
#
# Note on the sample. The earlier version, which read the workbook, got 9 participants because
# the EEG columns of 008 were empty, and nothing in the pipeline output justifies that gap (008
# has both resting runs surviving and the same channel count as the others). The default uses
# every participant with complete data, which gives 10. To apply the win-stay criterion of
# preregistration 35.3 as well, pass --exclusions prereg (008 at .595 and 009 at .538 drop out, leaving 8).
import argparse, cohort
_ap=argparse.ArgumentParser()
_ap.add_argument('--exclusions',choices=['none','prereg'],default='none',
                 help='prereg applies the win-stay < .60 of preregistration 35.3')
_A=_ap.parse_args()
M=cohort.by_id(str(DATA))
if _A.exclusions=='prereg':
    M={p:d for p,d in M.items() if not d.get('excl_winstay')}
print(f"[subtype] exclusions '{_A.exclusions}' - {len(M)} candidates")
S=json.load(open(f'{DATA}/psd_all.json'))
POSTC=['O1','O2']

# -- individual alpha frequency --
IAF={}
for pid,d in S.items():
    f=np.array(d['f']); have=d['ch']
    ii=[have.index(c) for c in POSTC if c in have]
    if not ii: continue
    P=np.array(d['rest1'])
    if 'rest2' in d: P=(P+np.array(d['rest2']))/2
    m=(f>=2)&(f<=35)
    fm=FOOOF(peak_width_limits=[1,8],max_n_peaks=6,min_peak_height=.05,verbose=False)
    try: fm.fit(f[m],P[ii].mean(0)[m],[2,35])
    except Exception: continue
    if fm.r_squared_<.85: continue
    a=[p for p in fm.peak_params_ if 7<=p[0]<14]
    if a: IAF[pid]=float(max(a,key=lambda p:p[1])[0])

# -- state change --
TR={'001':[1.51,1.31,.79,.67,.55],'002':[1.86,1.87,1.68,1.67,1.66],
 '003':[1.33,1.31,1.28,1.30,1.30],'004':[1.19,1.22,1.83,1.85,1.84],
 '005':[.80,1.12,1.81,1.83,1.85],'006':[1.61,1.58,1.48,1.47,1.48],
 '007':[.44,.46,1.03,1.05,1.04],'008':[.43,.47,1.44,1.45,1.43],
 '009':[1.35,1.34,1.30,1.29,1.30],'012':[1.20,1.20,.53,.55,.54],
 '014':[1.60,1.60,1.60,1.60,1.60],'016':[1.70,1.69,1.66,1.67,1.66]}
DCH={p: float(np.mean(v[2:])-np.mean(v[:2])) for p,v in TR.items()}

FEAT=[('front_exp','aperiodic slope','EEG - direction axis'),
      ('front_alpha','alpha power','EEG - performance axis'),
      ('iaf','individual alpha frequency','EEG - slowing'),
      ('dexp','rest-to-task slope change','EEG - state flexibility'),
      ('lose_shift','lose-shift','behaviour - direction axis'),
      ('accuracy','accuracy','behaviour - performance axis')]

rows=[]
for pid in sorted(M):
    d=M[pid]; r={'id':pid}
    for k in ['front_exp','front_alpha','lose_shift','accuracy']:
        r[k]=d.get(k)
    r['iaf']=IAF.get(pid); r['dexp']=DCH.get(pid)
    if sum(1 for k,_,_ in FEAT if r.get(k) is None)==0: rows.append(r)
pids=[r['id'] for r in rows]
X=np.array([[r[k] for k,_,_ in FEAT] for r in rows],float)
print(f"participants with complete data: {len(pids)}: {pids}\n")

Xz=(X-X.mean(0))/X.std(0,ddof=0)
print("standardised profiles (z)")
print(f"{'ID':>5}" + "".join(f"{n[:7]:>10}" for _,n,_ in FEAT))
for i,p in enumerate(pids):
    print(f"{p:>5}" + "".join(f"{Xz[i,j]:>+10.2f}" for j in range(len(FEAT))))

# -- correlation between indices (duplication check) --
print("\ncorrelation between indices")
R=np.corrcoef(Xz.T)
print("        " + "".join(f"{n[:7]:>9}" for _,n,_ in FEAT))
for i,(_,n,_) in enumerate(FEAT):
    print(f"{n[:7]:>7} " + "".join(f"{R[i,j]:>9.2f}" for j in range(len(FEAT))))
ev=np.linalg.eigvalsh(R)[::-1]
print(f"\neffective dimensions {(ev.sum()**2)/(ev**2).sum():.2f} / {len(FEAT)}  "
      f"- first component {ev[0]/ev.sum()*100:.0f}%")

# -- hierarchical clustering --
D=pdist(Xz,metric='euclidean')
Z=linkage(D,method='average')
print("\n"+"="*72); print(" hierarchical clustering"); print("="*72)
from scipy.cluster.hierarchy import cophenet
c,_=cophenet(Z,D)
print(f"  cophenetic correlation {c:.3f}  (closer to 1 means the dendrogram reflects the distances better)")
for k in [2,3,4]:
    lab=fcluster(Z,k,criterion='maxclust')
    sizes=[int((lab==i).sum()) for i in range(1,k+1)]
    print(f"  k={k}: sizes {sizes}  " +
          " | ".join("cluster%d {%s}"%(i,', '.join(p for p,l in zip(pids,lab) if l==i))
                     for i in range(1,k+1)))

# -- bootstrap stability --
print("\n"+"="*72); print(" bootstrap stability (1000 resamples of the indices, k=2)"); print("="*72)
rng=np.random.default_rng(0); n=len(pids); co=np.zeros((n,n)); B=1000
for _ in range(B):
    sel=rng.choice(len(FEAT),len(FEAT),replace=True)
    Xb=Xz[:,sel]
    lab=fcluster(linkage(pdist(Xb),'average'),2,criterion='maxclust')
    for i in range(n):
        for j in range(n):
            if lab[i]==lab[j]: co[i,j]+=1
co/=B
lab2=fcluster(Z,2,criterion='maxclust')
print(f"{'':>5}" + "".join(f"{p:>6}" for p in pids))
for i,p in enumerate(pids):
    print(f"{p:>5}" + "".join(f"{co[i,j]*100:>6.0f}" for j in range(n)))
within=[];between=[]
for i in range(n):
    for j in range(i+1,n):
        (within if lab2[i]==lab2[j] else between).append(co[i,j])
print(f"\n  mean co-assignment for same-cluster pairs {np.mean(within)*100:.0f}%")
print(f"  mean co-assignment for different-cluster pairs {np.mean(between)*100:.0f}%")
print(f"  separation {(np.mean(within)-np.mean(between))*100:.0f} percentage points")

json.dump(dict(pids=pids, X=X.tolist(), Xz=Xz.tolist(),
               feat=[[k,n,g] for k,n,g in FEAT],
               Z=Z.tolist(), co=co.tolist(),
               lab2=lab2.tolist(),
               lab3=fcluster(Z,3,criterion='maxclust').tolist()),
          open(f'{DATA}/subtype.json','w'))
print("\nwritten")
