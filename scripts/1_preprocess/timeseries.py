"""
timeseries.py - event-locked time-resolved analysis

Markers
  30 fixation, 31 stimulus, 32 response, 33 feedback X = unrewarded, 34 feedback O = rewarded

For each trial the time course of per-band instantaneous power (Hilbert) and of frontal-
occipital PLV is produced, then averaged under two alignments, stimulus-locked and feedback-locked.
Conditions: unrewarded vs rewarded, and stay vs switch on the following trial
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW

import numpy as np, glob, csv, json, os, warnings, math
import mne
from scipy.signal import hilbert, butter, filtfilt
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")

CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FRONT=['AF3','AF4','F3','F4']; POST=['O1','O2']
BANDS={'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
RAW=str(RAW)
PRE, POST_T = 0.6, 1.6          # before and after the event (s)

def find(pid,pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}',recursive=True)
                      if os.path.basename(p).startswith(pid)))
def bp(x,lo,hi,fs,o=4):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=-1)

OUT={}
for pid in [f'{i:03d}' for i in range(1,17)]:
    edfs=[p for p in find(pid,'*.edf') if not p.endswith('.md.edf')]
    mks=find(pid,'*_intervalMarker.csv')
    behp=glob.glob(f'{RAW}/**/*_{pid}_behav.csv',recursive=True)
    if not(edfs and mks and behp): continue
    beh=list(csv.DictReader(open(behp[0],encoding='utf-8-sig')))
    M=[(int(r['marker_value']),float(r['latency']))
       for r in csv.DictReader(open(mks[0],encoding='utf-8-sig'))]
    stim=sorted(t for v,t in M if v==31)
    # fbC / fbE keep their historical names. Marker 33 is feedback X (unrewarded) and 34 is
    # feedback O (rewarded) - see the marker table in task/prl_task.py.
    fbC =sorted(t for v,t in M if v==33)
    fbE =sorted(t for v,t in M if v==34)
    if len(stim)<20: continue

    raw=mne.io.read_raw_edf(edfs[0],preload=True)
    have=[c for c in CH if c in raw.ch_names]
    raw.pick(have).filter(1.,45.,verbose=False).notch_filter(60.,verbose=False)
    fs=raw.info['sfreq']; D0=raw.get_data()*1e6

    # exclude bad channels
    npts=int((PRE+POST_T)*fs); ptp=[]
    for t in stim:
        s=D0[:,int((t-PRE)*fs):int((t-PRE)*fs)+npts]
        if s.shape[1]==npts: ptp.append(s.max(1)-s.min(1))
    P=np.array(ptp); med=np.median(P,axis=0); g=np.median(med)
    bad=[have[i] for i in range(len(have)) if med[i]>3*g or med[i]<0.2*g]
    good=[c for c in have if c not in bad]
    if len(good)<8: continue
    raw.pick(good); D=raw.get_data()*1e6
    idx={c:i for i,c in enumerate(good)}
    fi=[idx[c] for c in FRONT if c in idx]; pi=[idx[c] for c in POST if c in idx]

    # analytic signal per band
    AN={}
    for b,(lo,hi) in BANDS.items():
        AN[b]=hilbert(bp(D,lo,hi,fs),axis=-1)

    def epoch(ts):
        """list of event times -> dict of (n_event, n_time)"""
        n=int((PRE+POST_T)*fs); res={k:[] for k in
              ['f_theta','f_alpha','f_beta','p_alpha','plv_fp']}
        keep=[]
        for t in ts:
            a=int((t-PRE)*fs); b_=a+n
            if a<0 or b_>D.shape[1]: keep.append(False); continue
            seg=D[:,a:b_]
            if (seg.max(1)-seg.min(1)).max()>200: keep.append(False); continue
            keep.append(True)
            for bn,key in [('theta','f_theta'),('alpha','f_alpha'),('beta','f_beta')]:
                amp=np.abs(AN[bn][:,a:b_])**2
                res[key].append(amp[fi].mean(0) if fi else np.full(n,np.nan))
            amp=np.abs(AN['alpha'][:,a:b_])**2
            res['p_alpha'].append(amp[pi].mean(0) if pi else np.full(n,np.nan))
            # frontal-occipital PLV: 0.4 s moving window
            ph=np.angle(AN['alpha'][:,a:b_]); w=int(.4*fs); pv=np.full(n,np.nan)
            for x in range(0,n-w+1):
                vals=[abs(np.mean(np.exp(1j*(ph[u,x:x+w]-ph[v,x:x+w]))))
                      for u in fi for v in pi]
                pv[x+w//2]=np.mean(vals) if vals else np.nan
            res['plv_fp'].append(pv)
        return {k:(np.array(v) if v else np.zeros((0,n))) for k,v in res.items()}, keep

    E_stim,_=epoch(stim)
    E_C,_=epoch(fbC); E_E,_=epoch(fbE)
    OUT[pid]=dict(
        n_time=int((PRE+POST_T)*fs), fs=float(fs),
        stim={k:v.mean(0).tolist() if len(v) else [] for k,v in E_stim.items()},
        fbC ={k:v.mean(0).tolist() if len(v) else [] for k,v in E_C.items()},
        fbE ={k:v.mean(0).tolist() if len(v) else [] for k,v in E_E.items()},
        n_stim=len(E_stim['f_alpha']), n_C=len(E_C['f_alpha']), n_E=len(E_E['f_alpha']))
    print(f"  {pid}: stimulus {OUT[pid]['n_stim']} - unrewarded feedback {OUT[pid]['n_C']} "
          f"- rewarded feedback {OUT[pid]['n_E']}  (good channels {len(good)})")

json.dump(OUT, open(f'{DATA}/timeseries.json','w'))
print(f"\n{len(OUT)} participants")
