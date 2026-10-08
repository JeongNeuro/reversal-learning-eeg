"""
timescale_checks.py - follow-up checks on the intrinsic timescale
 1. stability across bin sizes (40 / 50 / 60 ms)
 2. tau by band (broadband, alpha, theta)
 3. the hierarchy re-examined with the temporal electrodes excluded
 4. reproducibility across the two resting runs
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import numpy as np, glob, csv, json, os, warnings, math
import mne
from scipy.signal import hilbert, butter, filtfilt
from scipy.optimize import curve_fit
from scipy import stats
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")

CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
AP={'AF3':.66,'AF4':.66,'F7':.40,'F3':.42,'F4':.42,'F8':.40,'FC5':.16,'FC6':.16,
    'T7':0,'T8':0,'P7':-.48,'P8':-.48,'O1':-.76,'O2':-.76}
TEMP=['T7','T8','FC5','FC6']; FRONT=['AF3','AF4','F3','F4']
RAW=str(RAW); WIN=1.6
BANDS={'broad':(1,45),'alpha':(8,13),'theta':(4,8)}

def find(pid,pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}',recursive=True)
                      if os.path.basename(p).startswith(pid)))
def bp(x,lo,hi,fs,o=4):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=-1)
def decay(k,A,tau,B): return A*(np.exp(-k/tau)+B)

def fit_tau(X, BIN):
    """X: (epoch, bin) -> tau"""
    nb=X.shape[1]
    Xz=(X-X.mean(0))/(X.std(0)+1e-12)
    lags=[];vals=[]
    for k in range(1,nb):
        rs=[np.corrcoef(Xz[:,i],Xz[:,i+k])[0,1] for i in range(nb-k)]
        rs=[r for r in rs if not math.isnan(r)]
        if rs: lags.append(k*BIN); vals.append(np.mean(rs))
    if len(lags)<6: return None
    lags=np.array(lags); vals=np.array(vals); m=lags>=2*BIN
    try:
        p0=[max(vals[m][0],.05),.2,.05]
        popt,_=curve_fit(decay,lags[m],vals[m],p0=p0,
                         bounds=([0,.02,-.5],[3,3,1]),maxfev=8000)
        pred=decay(lags[m],*popt)
        r2=1-((vals[m]-pred)**2).sum()/(((vals[m]-vals[m].mean())**2).sum()+1e-12)
        return dict(tau=float(popt[1]),r2=float(r2)) if r2>=.5 else None
    except Exception: return None

RES={}
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
    fs=raw.info['sfreq']; D0=raw.get_data()*1e6; npts=int(WIN*fs)
    ptp=[]
    for t in stim:
        a=int((t-WIN)*fs)
        if a<0: continue
        s=D0[:,a:a+npts]
        if s.shape[1]==npts: ptp.append(s.max(1)-s.min(1))
    if len(ptp)<20: continue
    P=np.array(ptp); med=np.median(P,0); g=np.median(med)
    good=[have[i] for i in range(len(have)) if not(med[i]>3*g or med[i]<.2*g)]
    if len(good)<8: continue
    raw.pick(good); D=raw.get_data()*1e6
    ENV={b:np.abs(hilbert(bp(D,lo,hi,fs),axis=-1)) for b,(lo,hi) in BANDS.items()}

    # indices of the valid trials
    ok=[]
    for t in stim:
        a=int((t-WIN)*fs)
        if a<0 or a+npts>D.shape[1]: continue
        if (D[:,a:a+npts].max(1)-D[:,a:a+npts].min(1)).max()>200: continue
        ok.append(a)
    if len(ok)<25: continue

    rec={'good':good}
    # 1) bin size x 2) band
    for BIN in [.04,.05,.06]:
        nb=int(WIN/BIN); sb=int(BIN*fs)
        for band in BANDS:
            E=ENV[band]; taus={}
            for ci,c in enumerate(good):
                X=np.array([[E[ci,a+b*sb:a+(b+1)*sb].mean() for b in range(nb)]
                            for a in ok])
                r=fit_tau(X,BIN)
                if r: taus[c]=r['tau']
            rec[f'{band}_{int(BIN*1000)}']=taus

    # 4) the two resting runs
    for sv,ev,nm in ((10,11,'rest1'),(12,13,'rest2')):
        a_=[t for v,t in M if v==sv]; b_=[t for v,t in M if v==ev]
        if not(a_ and b_): continue
        s0=a_[-1]; e0=[t for t in b_ if t>s0]
        if not e0: continue
        s0=max(0,s0); e0=min(e0[0],raw.times[-1])
        BIN=.05; nb=int(WIN/BIN); sb=int(BIN*fs); step=int(WIN*fs)
        segs=[int(s0*fs)+i*step for i in range(int((e0-s0)/WIN))]
        segs=[a for a in segs if a+npts<=D.shape[1]]
        if len(segs)<20: continue
        E=ENV['broad']; taus={}
        for ci,c in enumerate(good):
            X=np.array([[E[ci,a+b*sb:a+(b+1)*sb].mean() for b in range(nb)]
                        for a in segs])
            r=fit_tau(X,BIN)
            if r: taus[c]=r['tau']
        rec[f'rest_{nm}']=taus
    RES[pid]=rec
    print(f"  {pid}: {len(good)} channels - {len(ok)} trials - "
          f"resting {'rest_rest1' in rec}/{'rest_rest2' in rec}")

json.dump(RES, open(f'{DATA}/timescale_checks.json','w'))

# ---------- summary ----------
def chanmed(key):
    d={}
    for pid,r in RES.items():
        for c,v in r.get(key,{}).items(): d.setdefault(c,[]).append(v)
    return {c:np.median(v)*1000 for c,v in d.items() if len(v)>=5}

print("\n"+"="*72); print(" 1. stability across bin sizes (broadband)"); print("="*72)
sets={}
for b in [40,50,60]:
    m=chanmed(f'broad_{b}'); sets[b]=m
    print(f"  {b} ms: {len(m)} channels - median tau {np.median(list(m.values())):.0f} ms")
common=set(sets[40])&set(sets[50])&set(sets[60])
if len(common)>=8:
    a=[sets[40][c] for c in common]; b_=[sets[50][c] for c in common]
    c_=[sets[60][c] for c in common]
    print(f"  40 vs 50 ms channel rank correlation rho = {stats.spearmanr(a,b_).statistic:+.3f}")
    print(f"  50 vs 60 ms channel rank correlation rho = {stats.spearmanr(b_,c_).statistic:+.3f}")

print("\n"+"="*72); print(" 2. tau by band (50 ms)"); print("="*72)
for band in BANDS:
    m=chanmed(f'{band}_50')
    if not m: continue
    v=list(m.values())
    r=stats.spearmanr([AP[c] for c in m],v)
    print(f"  {band:>6}: {len(m):>2} channels - tau {np.median(v):>5.0f} ms "
          f"(range {min(v):.0f}-{max(v):.0f}) - anterior-posterior position rho = {r.statistic:+.3f} "
          f"(p={r.pvalue:.3f})")

print("\n"+"="*72); print(" 3. the hierarchy with the temporal electrodes excluded"); print("="*72)
m=chanmed('broad_50')
for label,excl in [('all',[]),('temporal excluded',TEMP)]:
    sub={c:v for c,v in m.items() if c not in excl}
    if len(sub)<6: continue
    r=stats.spearmanr([AP[c] for c in sub],list(sub.values()))
    print(f"  {label:>20} ({len(sub)} channels): rho = {r.statistic:+.3f} (p={r.pvalue:.3f})")

print("\n"+"="*72); print(" 4. reproducibility across the two resting runs"); print("="*72)
pairs=[]
for pid,r in RES.items():
    t1=r.get('rest_rest1',{}); t2=r.get('rest_rest2',{})
    f1=[t1[c] for c in FRONT if c in t1]; f2=[t2[c] for c in FRONT if c in t2]
    if f1 and f2: pairs.append((pid,np.mean(f1)*1000,np.mean(f2)*1000))
if len(pairs)>=6:
    a=[x[1] for x in pairs]; b_=[x[2] for x in pairs]
    r=stats.pearsonr(a,b_)
    print(f"  frontal tau between runs: r = {r[0]:+.3f} (p={r[1]:.4f}, n={len(pairs)})")
    for pid,x,y in sorted(pairs,key=lambda z:z[1]):
        print(f"    {pid}: {x:>6.0f} → {y:>6.0f} ms")
else:
    print(f"  {len(pairs)} run pairs - cannot be computed")
