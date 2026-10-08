"""
persev_direct.py - stickiness indices computed without estimation, plus a cluster-profile comparison

Indices (all computed directly from the data, no fitting)
  - longest run of repeated choices
  - runs that carried across reversals (how many reversals passed without a change)
  - moving-window lose-shift trajectory
  - distribution of run lengths
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import numpy as np, glob, csv, json, math, warnings
from collections import Counter
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")

RAW=str(RAW)
BEH={}
for p in glob.glob(f'{RAW}/**/*_behav.csv', recursive=True):
    pid=p.split('_')[-2]
    if pid not in BEH: BEH[pid]=list(csv.DictReader(open(p,encoding='utf-8-sig')))
pids=sorted(BEH)

# The behavioural and EEG indices come from cohort.py, the demographics from an optional input.
# Without the demographics, those columns of the cluster comparison are left empty.
# The disability column is free text. Each rule below maps a label to the substrings that
# select it, matched case-insensitively in order. Override any of them with the environment
# variables named, as comma-separated substrings, when the column is in another language.
def _w(env, default):
    return [s.strip().lower() for s in os.environ.get(env, default).split(',') if s.strip()]
DTYPE_RULES = [('autism', _w('ASD_DISABILITY_AUTISM', 'autis')),
               ('multiple', _w('ASD_DISABILITY_MULTIPLE', 'multipl')),
               ('developmental', _w('ASD_DISABILITY_DEVELOPMENTAL', 'develop')),
               ('intellectual', _w('ASD_DISABILITY_INTELLECTUAL', 'intellect'))]
GRADE_RULES = [('grade 1', _w('ASD_DISABILITY_GRADE1', 'grade 1,level 1')),
               ('grade 3', _w('ASD_DISABILITY_GRADE3', 'grade 3,level 3'))]
DOWN_WORDS = _w('ASD_DISABILITY_DOWN', 'down,trisomy')
import cohort
M=cohort.by_id(str(DATA))
PT=cohort.demographics(str(DATA)) or {}
HAVE_DEMO=bool(PT)

def gi(r,k,d=None):
    try: return int(r[k])
    except (KeyError,ValueError,TypeError): return d

print("="*80)
print(" stickiness indices computed without estimation")
print("="*80)
IDX={}
for pid in pids:
    B=BEH[pid]; ch=[r.get('choice') for r in B]
    rev=[i for i,r in enumerate(B) if r.get('is_reversal_trial')=='1']
    if len(ch)<40: continue
    # runs of repeated choices
    runs=[]; s=0
    for i in range(1,len(ch)):
        if ch[i]==ch[i-1]: continue
        runs.append((s,i-1)); s=i
    runs.append((s,len(ch)-1))
    lens=[b-a+1 for a,b in runs]
    longest=max(lens); li=int(np.argmax(lens)); a,b=runs[li]
    # how many reversals the longest run crossed
    crossed=sum(1 for r in rev if a<r<=b)
    # the largest number of reversals crossed by any run
    maxcross=max((sum(1 for r in rev if x<r<=y) for x,y in runs), default=0)
    # trials from a reversal to the first switch
    tosw=[]
    for r in rev:
        k=None
        for j in range(r,min(r+40,len(ch))):
            if j>0 and ch[j]!=ch[j-1]: k=j-r; break
        tosw.append(k if k is not None else 40)
    IDX[pid]=dict(longest=int(longest), crossed=int(crossed), maxcross=int(maxcross),
                  med_run=float(np.median(lens)), n_runs=len(runs),
                  rep=float(np.mean([1 if ch[i]==ch[i-1] else 0
                                     for i in range(1,len(ch))])),
                  tosw=float(np.mean(tosw)), tosw_max=int(max(tosw)))
print(f"{'ID':>5}{'longest run':>13}{'rev crossed':>13}{'median len':>12}{'runs':>7}"
      f"{'repeat rate':>13}{'1st switch after rev':>22}")
for pid in sorted(IDX,key=lambda p:-IDX[p]['longest']):
    d=IDX[pid]
    print(f"{pid:>5}{d['longest']:>9}{d['maxcross']:>9}{d['med_run']:>9.1f}"
          f"{d['n_runs']:>8}{d['rep']*100:>8.1f}%{d['tosw']:>12.1f}")

# -- moving-window lose-shift trajectory --
W=15
TRAJ={}
for pid in pids:
    B=BEH[pid]; ch=[r.get('choice') for r in B]
    rw=[gi(r,'reward',gi(r,'correct')) for r in B]
    v=[]
    for i in range(len(ch)):
        lo=max(0,i-W//2); hi=min(len(ch)-1,i+W//2)
        num=den=0
        for j in range(lo,hi):
            if rw[j]==0 and j+1<len(ch):
                den+=1
                if ch[j+1]!=ch[j]: num+=1
        v.append(num/den if den>=3 else np.nan)
    v=np.array(v,float)
    m=~np.isnan(v)
    if m.sum()<len(v)*.6: continue
    v=np.interp(np.arange(len(v)),np.arange(len(v))[m],v[m])
    TRAJ[pid]=v[:120] if len(v)>=120 else np.pad(v,(0,120-len(v)),constant_values=v[-1])

tp=sorted(TRAJ); Xt=np.array([TRAJ[p] for p in tp])
print(f"\nlose-shift trajectory obtained for {len(tp)} participants")

print("\n"+"="*80)
print(" trajectory clusters (moving-window lose-shift, window of 15 trials)")
print("="*80)
Z=linkage(pdist(Xt),'ward')
from scipy.cluster.hierarchy import cophenet
coph,_=cophenet(Z,pdist(Xt)); print(f"  cophenetic correlation {coph:.3f}")
rng=np.random.default_rng(0)
for k in [2,3]:
    base=fcluster(Z,k,criterion='maxclust')
    ari=[]
    for _ in range(500):
        ix=rng.choice(len(tp),int(len(tp)*.8),replace=False)
        ari.append(adjusted_rand_score(base[ix],
                   fcluster(linkage(pdist(Xt[ix])),k,criterion='maxclust')))
    grp=' | '.join('{'+', '.join(p for p,l in zip(tp,base) if l==c)+'}'
                   for c in range(1,k+1))
    print(f"  k={k}: {grp}")
    print(f"        median resampled ARI {np.median(ari):+.3f} "
          f"[{np.percentile(ari,25):+.3f}, {np.percentile(ari,75):+.3f}]")
lab=fcluster(Z,3,criterion='maxclust')

print("\n"+"="*80)
print(" cluster profile comparison")
print("="*80)
if not HAVE_DEMO:
    print("  Note. The registration-type, Down, grade, age and medication columns are empty -")
    print(f"        they are participant data and are not in the repository. Add data/{cohort.DEMO_FILE} to fill them.")
# cohort.demographics() reads the CSV, so age and psychotropic_med arrive as strings.
# They are coerced here; without this np.mean() over the age column raises TypeError
# whenever data/demographics.csv is present.
def _num(v):
    try: return float(v)
    except (TypeError, ValueError): return None
rows=[]
for i,p in enumerate(tp):
    d=M.get(p,{}); q=PT.get(p,{}); dis=str(q.get('disability') or '')
    rows.append(dict(pid=p, grp=int(lab[i]),
        age=_num(q.get('age')), sex=q.get('sex'),
        med=_num(q.get('psychotropic_med')),
        dtype=next((_lb for _lb, ws in DTYPE_RULES if any(w in dis.lower() for w in ws)), 'unknown'),
        down=1 if any(w in dis.lower() for w in DOWN_WORDS) else 0,
        grade=next((_lb for _lb, ws in GRADE_RULES if any(w in dis.lower() for w in ws)), 'not recorded'),
        persev=d.get('n_persev'), regress=d.get('n_regress'),
        acc=d.get('accuracy'), ls=d.get('lose_shift'), ws=d.get('win_stay'),
        exp=d.get('front_exp'), alpha=d.get('front_alpha'),
        longest=IDX[p]['longest'], cross=IDX[p]['maxcross']))
print(f"{'ID':>5}{'cluster':>9}{'type':>16}{'Down':>6}{'grade':>13}{'age':>5}{'med':>5}"
      f"{'longest':>9}{'persev':>8}{'accuracy':>10}{'slope':>8}")
for r in sorted(rows,key=lambda z:(z['grp'],-z['longest'])):
    print(f"{r['pid']:>5}{r['grp']:>5}{r['dtype']:>9}{r['down']:>5}{r['grade']:>6}"
          f"{r['age'] or 0:>5}{r['med'] or 0:>5}{r['longest']:>9}"
          f"{r['persev'] or 0:>6}{r['acc'] or 0:>8.2f}"
          f"{r['exp'] if r['exp'] else float('nan'):>8.2f}")

print(f"\n{'variable':>22}" + "".join(f"{'cluster '+str(c):>12}" for c in [1,2,3]) + f"{'test':>20}")
def summarize(key,label,cat=False):
    vals=[[r[key] for r in rows if r['grp']==c and r[key] is not None] for c in [1,2,3]]
    if cat:
        cells=[]
        for v in vals:
            cnt=Counter(v); cells.append('/'.join(f'{k}:{n}' for k,n in cnt.most_common(3)) or '—')
        print(f"{label:>14}" + "".join(f"{c:>12}" for c in cells) + f"{'—':>20}")
        return
    if any(len(v)==0 for v in vals): return
    try:
        h=stats.kruskal(*vals)
        t=f"KW p={h.pvalue:.3f}"
    except Exception: t='—'
    print(f"{label:>14}" + "".join(f"{np.mean(v):>12.2f}" for v in vals) + f"{t:>20}")
for k,l in [('longest','longest run'),('cross','reversals crossed'),('persev','perseverative errors'),
            ('regress','regressive errors'),('acc','accuracy'),('ls','lose-shift'),
            ('ws','win-stay'),('exp','frontal slope'),('alpha','frontal alpha'),
            ('age','age')]:
    summarize(k,l)
for k,l in [('dtype','registration type'),('grade','disability grade'),('sex','sex')]:
    summarize(k,l,cat=True)
# binary variables
for k,l in [('down','Down syndrome'),('med','taking medication')]:
    a=[[r[k] for r in rows if r['grp']==c and r[k] is not None] for c in [1,2,3]]
    if any(len(v)==0 for v in a): continue
    print(f"{l:>14}" + "".join(f"{sum(v)}/{len(v):>10}" for v in a) + f"{'—':>20}")

json.dump(dict(idx=IDX, traj={p:TRAJ[p].tolist() for p in tp},
               tp=tp, lab=lab.tolist(),
               rows=[{k:(v if not isinstance(v,(np.integer,np.floating)) else float(v))
                      for k,v in r.items()} for r in rows]),
          open(f'{DATA}/persev.json','w'))
print("\nwritten")
