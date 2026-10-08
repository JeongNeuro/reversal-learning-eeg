"""
tfr.py - feedback-locked time-frequency decomposition
  - Morlet wavelets, 3-30 Hz over 20 log-spaced bins
  - unrewarded and rewarded feedback separately
  - dB against a baseline of -0.6 to -0.2 s (feedback-locked)

Note on the condition labels. Marker 33 is unrewarded and 34 is rewarded. For all 12
participants the count of marker 34 equals the number of reward==1 trials in the behavioural
record and also matches total_rewards in meta.json. An earlier version stored 33 as 'C'
(correct) and 34 as 'E' (error), which had the names the wrong way round. On an 80/20 schedule
a reward is always a correct choice, but unrewarded mixes errors with 'correct trials where
the reward was withheld', so rewarded/unrewarded is the accurate naming, not correct/error.

The per-condition baseline power is stored alongside, for diagnosis. The -0.6 to -0.2 s window
relative to feedback contains the response and the choice highlight (0.2 s), so if the
baseline itself differs by condition, the dB difference cannot be read as a post-feedback
effect. The stimulus-locked baseline fixed by preregistration 30 (-300 to -100 ms relative to the stimulus) is produced too.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW

import numpy as np, glob, csv, json, os, warnings
import mne
from scipy.signal import hilbert, butter, filtfilt
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")

CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FRONT=['AF3','AF4','F3','F4']
RAW=str(RAW)
FR=np.logspace(np.log10(3),np.log10(30),20)
PRE,POST=0.8,1.6
def find(pid,pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}',recursive=True)
                      if os.path.basename(p).startswith(pid)))

def morlet_power(x, fs, freqs, n_cycles=5):
    """x: (ch, t) -> (ch, f, t) power"""
    n=x.shape[-1]; out=np.zeros((x.shape[0],len(freqs),n))
    X=np.fft.fft(x,axis=-1); w=np.fft.fftfreq(n,1/fs)
    for k,f0 in enumerate(freqs):
        nc=n_cycles if f0>6 else 3
        sd=nc/(2*np.pi*f0)
        W=np.exp(-2*(np.pi*sd)**2*(w-f0)**2)*np.sqrt(2*np.pi*sd)
        out[:,k,:]=np.abs(np.fft.ifft(X*W,axis=-1))**2
    return out

OUT={}
for pid in [f'{i:03d}' for i in range(1,17)]:
    edfs=[p for p in find(pid,'*.edf') if not p.endswith('.md.edf')]
    mks=find(pid,'*_intervalMarker.csv')
    if not(edfs and mks): continue
    M=[(int(r['marker_value']),float(r['latency']))
       for r in csv.DictReader(open(mks[0],encoding='utf-8-sig'))]
    fbU=sorted(t for v,t in M if v==33)   # unrewarded
    fbR=sorted(t for v,t in M if v==34)   # rewarded
    stim=sorted(t for v,t in M if v==31)
    if len(fbU)<12 or len(fbR)<8: continue
    raw=mne.io.read_raw_edf(edfs[0],preload=True)
    have=[c for c in CH if c in raw.ch_names]
    raw.pick(have).filter(1.,45.,verbose=False).notch_filter(60.,verbose=False)
    fs=raw.info['sfreq']; D=raw.get_data()*1e6
    npts=int((PRE+POST)*fs)
    ptp=[]
    for t in stim:
        a=int(t*fs); s=D[:,a:a+npts]
        if s.shape[1]==npts: ptp.append(s.max(1)-s.min(1))
    if len(ptp)<20: continue
    P=np.array(ptp); med=np.median(P,0); g=np.median(med)
    keep=[i for i in range(len(have)) if not(med[i]>3*g or med[i]<.2*g)]
    good=[have[i] for i in keep]
    fi=[good.index(c) for c in FRONT if c in good]
    if len(good)<10 or not fi: continue
    D=D[keep]
    t_ax=np.arange(npts)/fs-PRE
    base=(t_ax>=-0.6)&(t_ax<=-0.2)
    rec={'ch':good,'t':t_ax.tolist(),'f':FR.tolist()}
    # stimulus-locked baseline (preregistration 30) - stimulus -300 to -100 ms
    st_lo, st_hi = int(-0.3*fs), int(-0.1*fs)
    for nm,ts in [('U',fbU),('R',fbR)]:
        acc=[]; n=0
        for t in ts:
            a=int((t-PRE)*fs)
            if a<0 or a+npts>D.shape[1]: continue
            seg=D[:,a:a+npts]
            if (seg.max(1)-seg.min(1)).max()>200: continue
            acc.append(morlet_power(seg,fs,FR)); n+=1
        if n<8: continue
        Pw=np.mean(acc,axis=0)                       # (ch,f,t)
        b=Pw[:,:,base].mean(axis=2,keepdims=True)
        dB=10*np.log10(np.maximum(Pw,1e-12)/np.maximum(b,1e-12))
        rec[nm]=dB[fi].mean(axis=0).tolist(); rec[f'n_{nm}']=n
        # For baseline diagnosis. dB takes the log per channel before averaging, so the
        # baseline and the raw power have to be kept in the same space (log per channel,
        # then average) for them to line up exactly with the dB above when the baseline is swapped.
        rec[f'basedb_{nm}']=(10*np.log10(np.maximum(b,1e-12)))[fi].mean(axis=0).ravel().tolist()
        rec[f'rawdb_{nm}']=(10*np.log10(np.maximum(Pw,1e-12)))[fi].mean(axis=0).tolist()
        rec[f'base_{nm}']=b[fi].mean(axis=0).ravel().tolist()
    # Stimulus-locked baseline - computed per condition at that trial's stimulus onset.
    #   Feedback follows the response, so the stimulus-to-feedback interval differs per trial.
    #   Each feedback trial is paired with the nearest preceding stimulus.
    for nm,ts in [('U',fbU),('R',fbR)]:
        acc=[]
        for t in ts:
            prev=[x for x in stim if x<t]
            if not prev: continue
            a=int(prev[-1]*fs)+st_lo; w=st_hi-st_lo
            if a<0 or a+w>D.shape[1]: continue
            seg=D[:,a:a+w]
            if (seg.max(1)-seg.min(1)).max()>200: continue
            acc.append(morlet_power(seg,fs,FR).mean(axis=2))
        if acc:
            SB=np.mean(acc,axis=0)
            rec[f'stimbase_{nm}']=SB[fi].mean(axis=0).tolist()
            rec[f'stimbasedb_{nm}']=(10*np.log10(np.maximum(SB,1e-12)))[fi].mean(axis=0).tolist()
            rec[f'n_stimbase_{nm}']=len(acc)

    if 'U' in rec and 'R' in rec:
        OUT[pid]=rec
        print(f"  {pid}: unrewarded {rec['n_U']} - rewarded {rec['n_R']}")

json.dump(OUT, open(f'{DATA}/tfr.json','w'))
print(f"\n{len(OUT)} participants - {len(FR)} frequency bins")
