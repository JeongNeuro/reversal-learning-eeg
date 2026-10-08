"""
phases.py - the four-phase split of Banerjee et al. (2020)

The windows are defined by performance level, not by trial number.
  LN  learning naive    in the first segment, before the criterion is reached
  LE  learning expert   in the first segment, after the criterion is reached
  RN  reversal naive    after a reversal, before the criterion is reached
  RE  reversal expert   after a reversal, after the criterion is reached
criterion = three correct in a row

Following the distinction of Wang et al. (2023), two things are examined separately.
  transient   the change in the short window right after a reversal   <- already tested in W1 (null)
  sustained   the change across the whole RN phase                     <- tested here
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW

import numpy as np, glob, csv, json, os, warnings, math
import mne
from scipy.signal import welch
from scipy import stats
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
from fooof import FOOOF

CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FRONT=['AF3','AF4','F3','F4']; POST=['O1','O2']
RAW=str(RAW)
def find(pid,pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}',recursive=True)
                      if os.path.basename(p).startswith(pid)))
def gi(r,k,d=None):
    try: return int(r[k])
    except (KeyError,ValueError,TypeError): return d

# ---------- 1. splitting the phases by behaviour ----------
PH={}
print("="*84); print(" four-phase split (criterion = three correct in a row)"); print("="*84)
print(f"{'ID':>5}{'LN':>6}{'LE':>6}{'RN':>6}{'RE':>6}{'unreached segments':>20}{'mean RN length':>16}")
BEH={}
for p in glob.glob(f'{RAW}/**/*_behav.csv',recursive=True):
    pid=p.split('_')[-2]
    if pid not in BEH: BEH[pid]=list(csv.DictReader(open(p,encoding='utf-8-sig')))
for pid in sorted(BEH):
    B=BEH[pid]; n=len(B)
    seg=[gi(r,'segment',0) for r in B]
    cor=[gi(r,'correct',0) for r in B]
    lab=['']*n; fails=0; rnlen=[]
    for s in sorted(set(seg)):
        ix=[i for i in range(n) if seg[i]==s]
        if not ix: continue
        # when the criterion was reached
        crit=None; run=0
        for k,i in enumerate(ix):
            run = run+1 if cor[i]==1 else 0
            if run>=3: crit=k; break
        naive='LN' if s==0 else 'RN'
        expert='LE' if s==0 else 'RE'
        if crit is None:
            for i in ix: lab[i]=naive
            fails+=1
            if s>0: rnlen.append(len(ix))
        else:
            for k,i in enumerate(ix):
                lab[i]= naive if k<=crit else expert
            if s>0: rnlen.append(crit+1)
    PH[pid]=lab
    c={k:lab.count(k) for k in ['LN','LE','RN','RE']}
    print(f"{pid:>5}{c['LN']:>6}{c['LE']:>6}{c['RN']:>6}{c['RE']:>6}"
          f"{fails:>12}{np.mean(rnlen) if rnlen else float('nan'):>12.1f}")

# ---------- 2. EEG by phase ----------
print("\n"+"="*84); print(" frontal aperiodic slope by phase"); print("="*84)
OUT={}
for pid in sorted(BEH):
    edfs=[p for p in find(pid,'*.edf') if not p.endswith('.md.edf')]
    mks=find(pid,'*_intervalMarker.csv')
    if not(edfs and mks) or pid not in PH: continue
    M=[(int(r['marker_value']),float(r['latency']))
       for r in csv.DictReader(open(mks[0],encoding='utf-8-sig'))]
    stim=sorted(t for v,t in M if v==31)
    lab=PH[pid]; nn=min(len(stim),len(lab))
    if nn<40: continue
    raw=mne.io.read_raw_edf(edfs[0],preload=True)
    have=[c for c in CH if c in raw.ch_names]
    raw.pick(have).filter(1.,45.,verbose=False).notch_filter(60.,verbose=False)
    fs=raw.info['sfreq']; nfft=int(2*fs); D0=raw.get_data()*1e6
    ptp=[]
    for t in stim[:nn]:
        a=int(t*fs); s=D0[:,a:a+nfft]
        if s.shape[1]==nfft: ptp.append(s.max(1)-s.min(1))
    if len(ptp)<20: continue
    Pm=np.array(ptp); med=np.median(Pm,0); g=np.median(med)
    good=[have[i] for i in range(len(have)) if not(med[i]>3*g or med[i]<.2*g)]
    if len(good)<8: continue
    raw.pick(good); D=raw.get_data()*1e6
    fi=[good.index(c) for c in FRONT if c in good]
    if not fi: continue
    res={}
    for ph in ['LN','LE','RN','RE']:
        ix=[i for i in range(nn) if lab[i]==ph]
        if len(ix)<8: continue
        ps=[]
        for i in ix:
            a=int(stim[i]*fs); s=D[:,a:a+nfft]
            if s.shape[1]<nfft: continue
            if (s.max(1)-s.min(1)).max()>200: continue
            f_,p_=welch(s,fs=fs,nperseg=nfft); ps.append(p_)
        if len(ps)<6: continue
        f_=welch(np.zeros((len(good),nfft)),fs=fs,nperseg=nfft)[0]
        Pw=np.mean(ps,axis=0); m=(f_>=2)&(f_<=40)
        fm=FOOOF(peak_width_limits=[1,8],max_n_peaks=6,min_peak_height=.05,verbose=False)
        try: fm.fit(f_[m],Pw[fi].mean(0)[m],[2,40])
        except Exception: continue
        if fm.r_squared_<.85: continue
        tot=Pw[:,(f_>=1)&(f_<=45)].sum(1)
        al=(Pw[:,(f_>=8)&(f_<13)].sum(1)/tot)[fi].mean()
        res[ph]=dict(exp=float(fm.aperiodic_params_[1]), alpha=float(al),
                     n=len(ps))
    if len(res)>=3: OUT[pid]=res

print(f"{'ID':>5}" + "".join(f"{p:>9}" for p in ['LN','LE','RN','RE'])
      + f"{'RN−RE':>9}{'RN−LE':>9}")
for pid in sorted(OUT):
    r=OUT[pid]
    cells=[f"{r[p]['exp']:.3f}" if p in r else '—' for p in ['LN','LE','RN','RE']]
    d1=r['RN']['exp']-r['RE']['exp'] if 'RN' in r and 'RE' in r else float('nan')
    d2=r['RN']['exp']-r['LE']['exp'] if 'RN' in r and 'LE' in r else float('nan')
    print(f"{pid:>5}" + "".join(f"{c:>9}" for c in cells)
          + f"{d1:>+9.3f}{d2:>+9.3f}")

print("\n"+"="*84); print(" test of sustained engagement (the distinction of Wang et al. 2023)"); print("="*84)
for a,b,lab in [('RN','RE','relearning vs expert after reversal'),
                ('RN','LE','relearning vs expert after learning'),
                ('LN','LE','early learning vs expert after learning')]:
    for key,kl in [('exp','aperiodic slope'),('alpha','alpha')]:
        pr=[(OUT[p][a][key],OUT[p][b][key]) for p in OUT if a in OUT[p] and b in OUT[p]]
        if len(pr)<6: continue
        d=[x-y for x,y in pr]
        w=stats.wilcoxon(d).pvalue
        print(f"  {lab:<24} {kl:<10} n={len(d):>2}  "
              f"median {np.median(d):+.4f}  higher side {sum(1 for x in d if x>0)}/{len(d)}  "
              f"p={w:.4f}")

# relation to behaviour
# The sample and the behavioural indices come from cohort.py alone (the dependency on the
# workbook's analysis sheet is removed - that file is participant data and is not published)
import cohort
MM=cohort.by_id(str(DATA))
print("\n"+"="*84); print(" length of the RN phase and the indices"); print("="*84)
rn=[(p,PH[p].count('RN')) for p in PH]
for key,lab in [('front_exp','resting frontal slope'),('lose_shift','lose-shift'),
                ('n_persev','perseverative errors')]:
    x=[];y=[]
    for p,c in rn:
        v=MM.get(p,{}).get(key)
        if v is not None and not MM.get(p,{}).get('excl_winstay'):
            x.append(c); y.append(v)
    if len(x)>=6:
        r=stats.spearmanr(x,y)
        print(f"  RN trial count x {lab:<24} rho = {r.statistic:+.3f}  p = {r.pvalue:.4f}  (n={len(x)})")

json.dump(dict(phase=PH, eeg=OUT), open(f'{DATA}/phases.json','w'))
print("\nwritten")
