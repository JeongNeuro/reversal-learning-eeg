"""
resolve_B.py - final computation of the outstanding items
Input: psd_all.json (resting runs 1 and 2 plus task, 12 participants), phases.json, tfr.json
       In tfr.json, U is unrewarded and R is rewarded (markers 33/34).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import json, numpy as np, math, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from fooof import FOOOF

S=json.load(open(f'{DATA}/psd_all.json'))
FRONT=['AF3','AF4','F3','F4']; POST=['O1','O2']
SEED, NB = 1, 10000          # the repository-wide convention


def boot_mean(d, seed=SEED, nb=NB):
    """Percentile bootstrap interval of the mean paired difference. The value reported in manuscript 3.4."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(nb)], [2.5, 97.5])

ALL=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
def L(t): print("\n"+"="*78+f"\n {t}\n"+"="*78)

def spec(pid,cond,chs):
    d=S[pid]; f=np.array(d['f']); have=d['ch']
    ii=[have.index(c) for c in chs if c in have]
    if not ii: return None,None
    if cond=='rest':
        P=np.array(d['rest1'])
        if 'rest2' in d: P=(P+np.array(d['rest2']))/2
    elif cond=='rest1': P=np.array(d['rest1'])
    else:
        if 'task' not in d: return None,None
        P=np.array(d['task'])
    return f, P[ii].mean(0)

def iaf_raw(f,p,lo,hi):
    """Peak of the raw spectrum"""
    m=(f>=lo)&(f<=hi)
    return float(f[m][np.argmax(p[m])])

def iaf_flat(f,p,lo,hi):
    """Peak after removing the aperiodic component"""
    m=(f>=2)&(f<=40)
    fm=FOOOF(peak_width_limits=[1,8],max_n_peaks=6,min_peak_height=.05,verbose=False)
    try: fm.fit(f[m],p[m],[2,40])
    except Exception: return np.nan
    if fm.r_squared_<.85: return np.nan
    flat=fm.power_spectrum-fm._ap_fit; ff=fm.freqs
    mm=(ff>=lo)&(ff<=hi)
    if not mm.any(): return np.nan
    return float(ff[mm][np.argmax(flat[mm])])

def iaf_fooof(f,p,lo,hi):
    """Centre frequency of the FOOOF peak (largest amplitude)"""
    m=(f>=2)&(f<=40)
    fm=FOOOF(peak_width_limits=[1,8],max_n_peaks=6,min_peak_height=.05,verbose=False)
    try: fm.fit(f[m],p[m],[2,40])
    except Exception: return np.nan,np.nan
    if fm.r_squared_<.85: return np.nan,np.nan
    a=[q for q in fm.peak_params_ if lo<=q[0]<=hi]
    if not a: return np.nan,np.nan
    b=max(a,key=lambda q:q[1])
    return float(b[0]), float(b[1])

def pk_raw(f,p,lo,hi):
    m=(f>=lo)&(f<=hi)
    return float(np.log10(p[m].max()))

# ══════════ B-1 ══════════
L("B-1. individual alpha frequency - rest vs task (by definition)")
pids=sorted(S)
print(f"{'definition':>42}{'n':>4}{'rest':>9}{'task':>9}{'change':>9}"
      f"{'95% CI':>20}{'Wilcoxon P':>13}")
CASES=[('raw spectrum peak - frontal - 7-14',iaf_raw,FRONT,7,14),
       ('raw spectrum peak - frontal - 8-13',iaf_raw,FRONT,8,13),
       ('raw spectrum peak - occipital - 7-14',iaf_raw,POST,7,14),
       ('raw spectrum peak - all channels - 7-14',iaf_raw,ALL,7,14),
       ('flat spectrum peak - frontal - 7-14',iaf_flat,FRONT,7,14),
       ('flat spectrum peak - occipital - 7-14',iaf_flat,POST,7,14),
       ('FOOOF peak - frontal - 7-14',None,FRONT,7,14),
       ('FOOOF peak - occipital - 7-14',None,POST,7,14)]
for lab,fn,chs,lo,hi in CASES:
    a=[];b=[]
    for p in pids:
        f1,s1=spec(p,'rest',chs); f2,s2=spec(p,'task',chs)
        if s1 is None or s2 is None: continue
        if fn is None:
            v1,_=iaf_fooof(f1,s1,lo,hi); v2,_=iaf_fooof(f2,s2,lo,hi)
        else:
            v1=fn(f1,s1,lo,hi); v2=fn(f2,s2,lo,hi)
        if np.isnan(v1) or np.isnan(v2): continue
        a.append(v1); b.append(v2)
    if len(a)<5: print(f"{lab:>34}{len(a):>4}{'—':>9}"); continue
    d=np.array(b)-np.array(a)
    P=stats.wilcoxon(d).pvalue if np.any(d!=0) else 1.0
    lo,hi=boot_mean(d)
    print(f"{lab:>34}{len(a):>4}{np.mean(a):>9.2f}{np.mean(b):>9.2f}{np.mean(d):>+9.2f}"
          f"{'[%+.2f, %+.2f]'%(lo,hi):>20}{P:>13.4f}")

# ══════════ B-2 ══════════
L("B-2. alpha peak power - rest vs task")
print(f"{'definition':>42}{'n':>4}{'rest':>9}{'task':>9}{'change':>9}"
      f"{'95% CI':>22}{'decrease':>10}{'Wilcoxon P':>13}")
for lab,mode,chs in [('raw spectrum peak log power - frontal','raw',FRONT),
                     ('raw spectrum peak log power - occipital','raw',POST),
                     ('FOOOF peak amplitude - frontal','fooof',FRONT),
                     ('FOOOF peak amplitude - occipital','fooof',POST)]:
    a=[];b=[]
    for p in pids:
        f1,s1=spec(p,'rest',chs); f2,s2=spec(p,'task',chs)
        if s1 is None or s2 is None: continue
        if mode=='raw': v1=pk_raw(f1,s1,7,14); v2=pk_raw(f2,s2,7,14)
        else:
            _,v1=iaf_fooof(f1,s1,7,14); _,v2=iaf_fooof(f2,s2,7,14)
        if np.isnan(v1) or np.isnan(v2): continue
        a.append(v1); b.append(v2)
    if len(a)<5: print(f"{lab:>34}{len(a):>4}{'—':>9}"); continue
    d=np.array(b)-np.array(a)
    P=stats.wilcoxon(d).pvalue
    lo,hi=boot_mean(d)
    print(f"{lab:>34}{len(a):>4}{np.mean(a):>9.3f}{np.mean(b):>9.3f}{np.mean(d):>+9.3f}"
          f"{'[%+.3f, %+.3f]'%(lo,hi):>22}"
          f"{sum(1 for x in d if x<0):>5}/{len(d)}{P:>13.4f}")

# ---------- B-3 recheck ----------
L("B-3. occipital individual alpha frequency (resting state)")
for lab,fn in [('raw spectrum peak 7-14',lambda f,p:iaf_raw(f,p,7,14)),
               ('raw spectrum peak 8-13',lambda f,p:iaf_raw(f,p,8,13)),
               ('flat spectrum peak 7-14',lambda f,p:iaf_flat(f,p,7,14)),
               ('FOOOF peak 7-14',lambda f,p:iaf_fooof(f,p,7,14)[0])]:
    v=[]
    for p in pids:
        f,s=spec(p,'rest',POST)
        if s is None: continue
        x=fn(f,s)
        if not np.isnan(x): v.append(x)
    if v: print(f"  {lab:>26}: mean {np.mean(v):.2f} Hz (SD {np.std(v,ddof=1):.2f}), n={len(v)}"
                f" - range {min(v):.1f}-{max(v):.1f}")

# ══════════ B-7 ══════════
L("B-7. comparison by learning phase")
ph=json.load(open(f'{DATA}/phases.json'))
E=ph['eeg']
print(f"  EEG participants {len(E)}: {', '.join(sorted(E))}")
ks=set()
for v in E.values(): ks|=set(v)
print(f"  phases: {sorted(ks)}")
full=[p for p in E if all(k in E[p] for k in ['LE','RN','RE'])]
print(f"  with all three phases: {len(full)} - {', '.join(sorted(full))}\n")
for met in ['exp','alpha']:
    M=np.array([[E[p][k][met] for k in ['LE','RN','RE']] for p in full])
    fr=stats.friedmanchisquare(*M.T)
    print(f"  [{met}] LE {M[:,0].mean():.3f} · RN {M[:,1].mean():.3f} · RE {M[:,2].mean():.3f}")
    print(f"        Friedman χ² = {fr.statistic:.3f}, P = {fr.pvalue:.4f}, n = {len(M)}")
    for i,j,lab in [(0,1,'LE-RN'),(1,2,'RN-RE'),(0,2,'LE-RE')]:
        d=M[:,j]-M[:,i]
        if np.any(d!=0):
            print(f"        {lab}: Wilcoxon P = {stats.wilcoxon(d).pvalue:.4f}")

# ══════════ B-6 ══════════
L("B-6. frontal alpha on rewarded vs unrewarded trials (tfr.json)")
T=json.load(open(f'{DATA}/tfr.json'))
print(f"  participants {len(T)}: {', '.join(sorted(T))}")
p0=list(T)[0]; f=np.array(T[p0]['f']); t=np.array(T[p0]['t'])
print(f"  frequency {f[0]:.1f}-{f[-1]:.1f} Hz ({len(f)} bins) - time {t[0]:.2f}-{t[-1]:.2f} s")
print(f"\n{'window':>20}{'n':>4}{'unrewarded':>12}{'rewarded':>10}{'lower':>8}{'Wilcoxon P':>13}")
for lo,hi,flo,fhi,lab in [(0,1,8,13,'0-1s - 8-13Hz'),(0,.5,8,13,'0-0.5s - 8-13Hz'),
                          (.2,.8,8,13,'0.2–0.8s · 8–13Hz'),(0,1,8,12,'0–1s · 8–12Hz'),
                          (.3,1.2,8,13,'0.3–1.2s · 8–13Hz')]:
    a=[];b=[]
    for p in sorted(T):
        C=np.array(T[p]['U']); Er=np.array(T[p]['R'])   # U unrewarded, R rewarded
        fm=(f>=flo)&(f<=fhi); tm=(t>=lo)&(t<=hi)
        a.append(C[np.ix_(fm,tm)].mean()); b.append(Er[np.ix_(fm,tm)].mean())
    d=np.array(b)-np.array(a)
    P=stats.wilcoxon(d).pvalue
    print(f"{lab:>18}{len(a):>4}{np.mean(a):>9.3f}{np.mean(b):>9.3f}"
          f"{sum(1 for x in d if x<0):>5}/{len(d)}{P:>13.4f}")
