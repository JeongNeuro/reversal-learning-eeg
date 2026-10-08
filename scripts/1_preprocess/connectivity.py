"""
connectivity.py - PLV per channel pair and per frequency band
  - resting state and task windows separately
  - 2-30 Hz in 15 bands of 2 Hz width
  - node degree = number of edges above the threshold
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW

import numpy as np, glob, csv, json, os, warnings
import mne
from scipy.signal import hilbert, butter, filtfilt
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")

CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
RAW=str(RAW)
FREQS=[(f, f+2) for f in range(2, 30, 2)]      # 14 bands
def find(pid,pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}',recursive=True)
                      if os.path.basename(p).startswith(pid)))
def bp(x,lo,hi,fs,o=4):
    b,a=butter(o,[lo/(fs/2),min(hi,fs/2-1)/(fs/2)],btype='band')
    return filtfilt(b,a,x,axis=-1)

def plv_epochs(D, fs, starts, npts, lo, hi):
    """Mean channel-pair PLV over the given windows"""
    f=bp(D,lo,hi,fs); ph=np.angle(hilbert(f,axis=-1))
    n=D.shape[0]; acc=np.zeros((n,n)); cnt=0
    for a in starts:
        if a<0 or a+npts>D.shape[1]: continue
        p=ph[:,a:a+npts]
        for i in range(n):
            for j in range(i+1,n):
                v=abs(np.mean(np.exp(1j*(p[i]-p[j]))))
                acc[i,j]+=v; acc[j,i]+=v
        cnt+=1
    return acc/cnt if cnt else None, cnt

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
    if len(have)<12: continue
    raw.pick(have).filter(1.,45.,verbose=False).notch_filter(60.,verbose=False)
    fs=raw.info['sfreq']; npts=int(2*fs); D=raw.get_data()*1e6
    # bad channels
    ptp=[]
    for t in stim:
        a=int(t*fs); s=D[:,a:a+npts]
        if s.shape[1]==npts: ptp.append(s.max(1)-s.min(1))
    if len(ptp)<20: continue
    P=np.array(ptp); med=np.median(P,0); g=np.median(med)
    bad=[i for i in range(len(have)) if med[i]>3*g or med[i]<.2*g]
    keep=[i for i in range(len(have)) if i not in bad]
    good=[have[i] for i in keep]
    if len(good)<10: continue
    D=D[keep]
    # resting windows
    rest=[]
    for sv,ev in ((10,11),(12,13)):
        a_=[t for v,t in M if v==sv]; b_=[t for v,t in M if v==ev]
        if not(a_ and b_): continue
        s0=a_[-1]; e0=[t for t in b_ if t>s0]
        if not e0: continue
        s0=max(0,s0); e0=min(e0[0],raw.times[-1])
        step=npts
        rest+= [int(s0*fs)+k*step for k in range(int((e0-s0)*fs)//step)]
    task=[int(t*fs) for t in stim]
    # drop the artefact windows
    def clean(sts):
        out=[]
        for a in sts:
            if a<0 or a+npts>D.shape[1]: continue
            s=D[:,a:a+npts]
            if (s.max(1)-s.min(1)).max()<=200: out.append(a)
        return out
    rest=clean(rest); task=clean(task)
    if len(rest)<15 or len(task)<30: continue
    rec={'ch':good,'n_rest':len(rest),'n_task':len(task),'rest':{},'task':{}}
    for lo,hi in FREQS:
        for nm,sts in [('rest',rest),('task',task)]:
            Pm,c=plv_epochs(D,fs,sts,npts,lo,hi)
            if Pm is not None: rec[nm][f'{lo}-{hi}']=Pm.tolist()
    OUT[pid]=rec
    print(f"  {pid}: {len(good)} channels - {len(rest)} resting and {len(task)} task windows")

json.dump(OUT, open(f'{DATA}/conn.json','w'))
print(f"\n{len(OUT)} participants - {len(FREQS)} frequency bands")
