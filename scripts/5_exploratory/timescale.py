"""
timescale.py - the intrinsic timescale (the approach of Murray et al. 2014, plus the aperiodic knee)

Method 1 (Murray)
  - the pre-trial window (feedback +0.7 s to just before the stimulus) is split into 50 ms bins
  - the band amplitudes of bins i and j are correlated across trials (Pearson)
  - the mean correlation per time lag is fitted with R(kD) = A[exp(-kD/tau) + B]
  - tau = the intrinsic timescale

Method 2 (Gao et al. 2020)
  - the resting spectrum is fitted with a knee model
  - tau = 1/(2*pi*f_knee)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW

import numpy as np, glob, csv, json, os, warnings, math
import mne
from scipy.signal import hilbert, butter, filtfilt
from scipy.optimize import curve_fit
from scipy import stats
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
from fooof import FOOOF

CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
# anterior-posterior axis coordinate (anterior is +)
AP={'AF3':.66,'AF4':.66,'F7':.40,'F3':.42,'F4':.42,'F8':.40,
    'FC5':.16,'FC6':.16,'T7':0,'T8':0,'P7':-.48,'P8':-.48,'O1':-.76,'O2':-.76}
RAW=str(RAW); BIN=0.05; WIN=1.6      # 50 ms bins, a 1.6 s pre-trial window

def find(pid,pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}',recursive=True)
                      if os.path.basename(p).startswith(pid)))
def bp(x,lo,hi,fs,o=4):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=-1)
def decay(k, A, tau, B):
    return A*(np.exp(-k/tau) + B)

OUT={}
for pid in [f'{i:03d}' for i in range(1,17)]:
    edfs=[p for p in find(pid,'*.edf') if not p.endswith('.md.edf')]
    mks=find(pid,'*_intervalMarker.csv')
    if not(edfs and mks): continue
    M=[(int(r['marker_value']),float(r['latency']))
       for r in csv.DictReader(open(mks[0],encoding='utf-8-sig'))]
    stim=sorted(t for v,t in M if v==31)
    if len(stim)<30: continue
    raw=mne.io.read_raw_edf(edfs[0],preload=True)
    have=[c for c in CH if c in raw.ch_names]
    raw.pick(have).filter(1.,45.,verbose=False).notch_filter(60.,verbose=False)
    fs=raw.info['sfreq']; D0=raw.get_data()*1e6
    npts=int(WIN*fs)
    # bad channels
    ptp=[]
    for t in stim:
        a=int((t-WIN)*fs)
        if a<0: continue
        s=D0[:,a:a+npts]
        if s.shape[1]==npts: ptp.append(s.max(1)-s.min(1))
    if len(ptp)<20: continue
    P=np.array(ptp); med=np.median(P,axis=0); g=np.median(med)
    bad=[have[i] for i in range(len(have)) if med[i]>3*g or med[i]<.2*g]
    good=[c for c in have if c not in bad]
    if len(good)<8: continue
    raw.pick(good); D=raw.get_data()*1e6

    # -- method 1: across-trial autocorrelation --
    nb=int(WIN/BIN); sb=int(BIN*fs)
    ENV={'broad':np.abs(hilbert(bp(D,1,45,fs),axis=-1)),
         'alpha':np.abs(hilbert(bp(D,8,13,fs),axis=-1))}
    res={}
    for bandname,A_ in ENV.items():
        # (trial, channel, bin) matrix
        Mx=[]
        for t in stim:
            a=int((t-WIN)*fs)
            if a<0 or a+npts>D.shape[1]: continue
            seg=D[:,a:a+npts]
            if (seg.max(1)-seg.min(1)).max()>200: continue
            e=A_[:,a:a+npts]
            Mx.append([[e[c,b*sb:(b+1)*sb].mean() for b in range(nb)]
                       for c in range(len(good))])
        Mx=np.array(Mx)                       # (trial, ch, bin)
        if len(Mx)<25: continue
        taus={}
        for ci,c in enumerate(good):
            X=Mx[:,ci,:]                       # (trial, bin)
            X=(X-X.mean(0))/(X.std(0)+1e-12)
            # mean correlation per lag
            lags=[]; vals=[]
            for k in range(1,nb):
                rs=[np.corrcoef(X[:,i],X[:,i+k])[0,1] for i in range(nb-k)]
                rs=[r for r in rs if not math.isnan(r)]
                if rs: lags.append(k*BIN); vals.append(np.mean(rs))
            lags=np.array(lags); vals=np.array(vals)
            # fit after dropping the first lag (adaptation)
            m=lags>=2*BIN
            try:
                p0=[vals[m][0] if vals[m][0]>0 else .1, .2, .05]
                popt,_=curve_fit(decay, lags[m], vals[m], p0=p0,
                                 bounds=([0,.02,-.5],[3,3,1]), maxfev=8000)
                pred=decay(lags[m],*popt)
                r2=1-((vals[m]-pred)**2).sum()/(((vals[m]-vals[m].mean())**2).sum()+1e-12)
                if r2>=.5: taus[c]=dict(tau=float(popt[1]),A=float(popt[0]),
                                        B=float(popt[2]),r2=float(r2))
            except Exception: pass
        res[bandname]=taus
    OUT[pid]=dict(good=good, ac=res)
    nb_ok=len(res.get('broad',{}))
    print(f"  {pid}: tau produced for {nb_ok} of {len(good)} channels")

json.dump(OUT, open(f'{DATA}/timescale.json','w'))

# -- summary --
print("\n"+"="*78)
print(" method 1: across-trial autocorrelation timescale (broadband 1-45 Hz)")
print("="*78)
allt={}
for pid,d in OUT.items():
    for c,v in d['ac'].get('broad',{}).items():
        allt.setdefault(c,[]).append(v['tau'])
print(f"{'channel':>9}{'n':>5}{'median tau (ms)':>17}{'A-P position':>14}")
rows=[]
for c in CH:
    if c not in allt: continue
    t=np.median(allt[c])*1000
    rows.append((c,len(allt[c]),t,AP[c]))
    print(f"{c:>5}{len(allt[c]):>5}{t:>13.0f}{AP[c]:>10.2f}")
if len(rows)>=8:
    r=stats.spearmanr([x[3] for x in rows],[x[2] for x in rows])
    print(f"\n  anterior-posterior position x tau:  rho = {r.statistic:+.3f}  (p = {r.pvalue:.4f})")
    print("  positive means a longer timescale further forward = the same direction as the Murray hierarchy")
