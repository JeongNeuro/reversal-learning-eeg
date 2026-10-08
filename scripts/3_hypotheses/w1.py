"""
w1.py - the preregistered within-participant hypothesis (W1)

Section 29.2 of the plan
  The aperiodic exponent in the window just after a reversal will be lower than in the stable window before it.
  - post-reversal window: 8 trials from the reversal trial
  - stable window: the 8 trials before the reversal
  - mixed-effects model (participant random intercept, reversal pair)

Reproducibility is also checked at window sizes of 6, 8 and 10 trials.
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

def fit_exp(seg, fs, nfft):
    """window -> frontal/occipital aperiodic exponent"""
    n=nfft; step=n//2; ps=[]
    if seg.shape[1]<n: return None
    for x in range(0,seg.shape[1]-n+1,step):
        ep=seg[:,x:x+n]
        if (ep.max(1)-ep.min(1)).max()>200: continue
        f_,p_=welch(ep,fs=fs,nperseg=n); ps.append(p_)
    if len(ps)<3: return None
    f_=welch(np.zeros((seg.shape[0],n)),fs=fs,nperseg=n)[0]
    P=np.mean(ps,axis=0); m=(f_>=2)&(f_<=40)
    return f_[m], P[:,m], len(ps)

OUT=[]
for pid in [f'{i:03d}' for i in range(1,17)]:
    edfs=[p for p in find(pid,'*.edf') if not p.endswith('.md.edf')]
    mks=find(pid,'*_intervalMarker.csv')
    behp=glob.glob(f'{RAW}/**/*_{pid}_behav.csv',recursive=True)
    if not(edfs and mks and behp): continue
    beh=list(csv.DictReader(open(behp[0],encoding='utf-8-sig')))
    M=[(int(r['marker_value']),float(r['latency']))
       for r in csv.DictReader(open(mks[0],encoding='utf-8-sig'))]
    stim=sorted(t for v,t in M if v==31)
    if len(stim)<30: continue
    # indices of the reversal trials
    rev=[i for i,b in enumerate(beh[:len(stim)]) if b.get('is_reversal_trial')=='1']
    if not rev: continue

    raw=mne.io.read_raw_edf(edfs[0],preload=True)
    have=[c for c in CH if c in raw.ch_names]
    raw.pick(have).filter(1.,45.,verbose=False).notch_filter(60.,verbose=False)
    fs=raw.info['sfreq']; nfft=int(2*fs)
    D0=raw.get_data()*1e6
    # bad channels
    ptp=[]
    for t in stim:
        a=int(t*fs); s=D0[:,a:a+nfft]
        if s.shape[1]==nfft: ptp.append(s.max(1)-s.min(1))
    if len(ptp)<20: continue
    P_=np.array(ptp); med=np.median(P_,0); g=np.median(med)
    good=[have[i] for i in range(len(have)) if not(med[i]>3*g or med[i]<.2*g)]
    if len(good)<8: continue
    raw.pick(good); D=raw.get_data()*1e6
    fi=[good.index(c) for c in FRONT if c in good]
    pi=[good.index(c) for c in POST if c in good]
    if not fi: continue

    for W in [6,8,10]:
        for r_ in rev:
            # stable window: the W trials before the reversal; post-reversal window: W trials from the reversal
            pre=[stim[j] for j in range(max(0,r_-W), r_) if j<len(stim)]
            post=[stim[j] for j in range(r_, min(r_+W,len(stim)))]
            if len(pre)<W*0.7 or len(post)<W*0.7: continue
            segs={}
            for nm,ts in [('pre',pre),('post',post)]:
                a=int(ts[0]*fs); b=int(min(ts[-1]+2.0, raw.times[-1])*fs)
                out=fit_exp(D[:,a:b], fs, nfft)
                if out is None: break
                f_,Pm,nep=out
                vals={}
                for rg,ix in [('front',fi),('post',pi)]:
                    if not ix: continue
                    fm=FOOOF(peak_width_limits=[1,8],max_n_peaks=6,
                             min_peak_height=.05,verbose=False)
                    try: fm.fit(f_,Pm[ix].mean(0),[2,40])
                    except Exception: continue
                    if fm.r_squared_<.85: continue
                    vals[rg]=float(fm.aperiodic_params_[1])
                if 'front' not in vals: break
                segs[nm]=(vals, nep)
            if len(segs)==2:
                OUT.append(dict(pid=pid, W=W, rev=int(r_),
                    pre_front=segs['pre'][0]['front'],
                    post_front=segs['post'][0]['front'],
                    pre_post=segs['pre'][0].get('post'),
                    post_post=segs['post'][0].get('post'),
                    n_pre=segs['pre'][1], n_post=segs['post'][1]))
    print(f"  {pid}: {len(rev)} reversals - {sum(1 for o in OUT if o['pid']==pid)} pairs")

json.dump(OUT, open(f'{DATA}/w1.json','w'))

print("\n"+"="*74)
print(" W1: just after versus just before a reversal - frontal aperiodic exponent")
print("="*74)
for W in [6,8,10]:
    sub=[o for o in OUT if o['W']==W]
    if not sub: continue
    d=[o['post_front']-o['pre_front'] for o in sub]
    pids=sorted(set(o['pid'] for o in sub))
    # test on the per-participant means (the independent unit is the participant)
    pm=[np.mean([o['post_front']-o['pre_front'] for o in sub if o['pid']==p])
        for p in pids]
    w=stats.wilcoxon(pm).pvalue if len(pm)>=6 else float('nan')
    t=stats.ttest_1samp(pm,0)
    print(f"\n[window {W} trials]  {len(sub)} pairs - {len(pids)} participants")
    print(f"  median pair-level change {np.median(d):+.4f}  (pairs that decreased {sum(1 for x in d if x<0)}/{len(d)})")
    print(f"  median participant-mean change {np.median(pm):+.4f}  "
          f"(participants who decreased {sum(1 for x in pm if x<0)}/{len(pm)})")
    print(f"  Wilcoxon p = {w:.4f} · t({len(pm)-1}) = {t.statistic:+.2f}, p = {t.pvalue:.4f}")
    print(f"  Cohen d = {np.mean(pm)/np.std(pm,ddof=1):+.3f}")
    # percentile bootstrap 95% interval (the basis for the manuscript's "the interval includes 0")
    _rng = np.random.default_rng(0)
    _bs = np.array([np.mean(_rng.choice(pm, len(pm), replace=True))
                    for _ in range(10000)])
    _lo, _hi = np.percentile(_bs, [2.5, 97.5])
    print(f"  mean change {np.mean(pm):+.4f}  95% CI [{_lo:+.4f}, {_hi:+.4f}]"
          f"{'  <- includes 0' if _lo <= 0 <= _hi else ''}")

# per participant (W=8)
sub=[o for o in OUT if o['W']==8]
pids=sorted(set(o['pid'] for o in sub))
print("\n"+"="*74); print(" by participant (window of 8 trials)"); print("="*74)
print(f"{'ID':>5}{'pairs':>7}{'before':>9}{'after':>9}{'change':>9}{'direction':>11}")
for p in pids:
    s=[o for o in sub if o['pid']==p]
    a=np.mean([o['pre_front'] for o in s]); b=np.mean([o['post_front'] for o in s])
    print(f"{p:>5}{len(s):>7}{a:>9.3f}{b:>9.3f}{b-a:>+9.3f}{'decrease' if b<a else 'increase':>11}")

# relation to the perseverative errors
# The sample and the behavioural indices come from cohort.py alone (the dependency on the
# workbook's analysis sheet is removed - that file is participant data and is not published)
import cohort
MM=cohort.by_id(str(DATA))
print("\n"+"="*74); print(" post-reversal change and behaviour"); print("="*74)
x=[];ys={'n_persev':[],'lose_shift':[],'sbi':[]}
ids=[]
for p in pids:
    s=[o for o in sub if o['pid']==p]
    dv=np.mean([o['post_front']-o['pre_front'] for o in s])
    d=MM.get(p,{})
    if d.get('n_persev') is None: continue
    x.append(dv); ids.append(p)
    for k in ys: ys[k].append(d[k])
for k,lab in [('n_persev','perseverative errors'),('lose_shift','lose-shift'),('sbi','switch-bias index')]:
    if len(x)>=6:
        r=stats.spearmanr(x,ys[k])
        print(f"  change x {lab:<22} rho = {r.statistic:+.3f}  p = {r.pvalue:.4f}  (n={len(x)})")
