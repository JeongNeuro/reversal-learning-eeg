"""
ts_stats.py - a second look at the time-series statistics
 - per-time-point Wilcoxon plus cluster-based permutation correction
 - individual curves per participant
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import json, numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from scipy import stats

FONT="Liberation Sans"
rcParams.update({"font.family":FONT,"font.sans-serif":[FONT],"font.size":7,
 "mathtext.fontset":"custom","mathtext.rm":FONT,"mathtext.it":f"{FONT}:italic",
 "mathtext.default":"regular","axes.linewidth":.7,"axes.edgecolor":"#333",
 "xtick.major.width":.7,"ytick.major.width":.7,"xtick.major.size":2.2,
 "ytick.major.size":2.2,"xtick.labelsize":6.2,"ytick.labelsize":6.2,
 "axes.labelsize":7,"legend.fontsize":6.4,"legend.frameon":False,
 "figure.dpi":200,"savefig.dpi":600,"savefig.bbox":"tight","savefig.pad_inches":.03})
EXP,RED,G3="#2F6FA8","#C0392B","#cfcfcf"
PRE=0.6
T=json.load(open(f'{DATA}/timeseries.json'))
PIDS=[p for p in sorted(T) if T[p]['n_stim']>=40 and T[p]['n_C']>=15 and T[p]['n_E']>=10]
fs=T[PIDS[0]]['fs']; n=T[PIDS[0]]['n_time']; t=np.arange(n)/fs-PRE

def zbase(v):
    b=(t>=-0.6)&(t<=-0.1); m,s=np.nanmean(v[b]),np.nanstd(v[b])
    return (v-m)/s if s>0 else v*0

MEAS=[('f_alpha','Frontal alpha'),('f_theta','Frontal theta'),
      ('plv_fp','Fronto-posterior PLV')]

def cluster_perm(C,E,nperm=2000,alpha=.05,seed=0):
    """Per-time-point t test, then the statistic sum of adjacent significant runs against the sign-permutation distribution"""
    rng=np.random.default_rng(seed); nsub,nt=C.shape
    Dd=C-E
    def tvec(Dm):
        # The standard error is divided by the number of valid participants at that time point.
        # Dividing by nsub where values are missing, as at the window edges, inflates t.
        k=(~np.isnan(Dm)).sum(0)
        m=np.nanmean(Dm,0); s=np.nanstd(Dm,0,ddof=1)
        return m/(s/np.sqrt(np.maximum(k,1))+1e-12)
    def clusters(mask,stat):
        out=[];st=None
        for i in range(nt):
            if mask[i] and st is None: st=i
            elif not mask[i] and st is not None:
                out.append((st,i,float(np.abs(stat[st:i]).sum()))); st=None
        if st is not None: out.append((st,nt,float(np.abs(stat[st:nt]).sum())))
        return out
    crit=stats.t.ppf(1-alpha/2, nsub-1)
    obs=tvec(Dd); obs_cl=clusters(np.abs(obs)>crit,obs)
    ps=2*(1-stats.t.cdf(np.abs(obs),nsub-1))
    if not obs_cl: return ps,[],obs
    null=np.zeros(nperm)
    for j in range(nperm):
        sg=rng.choice([-1.,1.],nsub)[:,None]
        st=tvec(Dd*sg); cl=clusters(np.abs(st)>crit,st)
        null[j]=max([c[2] for c in cl]) if cl else 0.
    res=[(a,b,m,float((null>=m).mean())) for a,b,m in obs_cl]
    return ps,res,obs

print("="*80)
print(" per-time-point test plus cluster-based permutation correction (feedback-locked, unrewarded vs rewarded)")
print("="*80)
STORE={}
for key,lab in MEAS:
    C=[];E=[]
    for pid in PIDS:
        # fbC / fbE are historical key names. fbC holds marker 33 (feedback X = unrewarded)
        # and fbE holds marker 34 (feedback O = rewarded); see the marker table in task/prl_task.py
        # and the note in scripts/1_preprocess/tfr.py.
        vc=np.array(T[pid]['fbC'][key],float); ve=np.array(T[pid]['fbE'][key],float)
        if len(vc)!=n or len(ve)!=n: continue
        if np.all(np.isnan(vc)) or np.all(np.isnan(ve)): continue
        C.append(zbase(vc)); E.append(zbase(ve))
    C=np.array(C); E=np.array(E)
    ps,cl,tv=cluster_perm(C,E,nperm=2000)
    STORE[key]=dict(C=C,E=E,ps=ps,cl=cl)
    sig=(ps<.05)
    print(f"\n[{lab}]  n = {len(C)}")
    print(f"   time points with p<.05: {sig.sum()}/{n}")
    if cl:
        for s,e,m,pv in cl:
            print(f"   cluster {t[s]:+.2f} to {t[min(e,n-1)]:+.2f} s  "
                  f"statistic sum {m:.1f}  permutation p = {pv:.4f}"
                  f"{'  <- significant' if pv<.05 else ''}")
    else:
        print("   no adjacent significant run")
    # the lowest-p time point
    i=int(np.nanargmin(ps))   # skips the NaN at the window edges
    print(f"   lowest p: p = {ps[i]:.4f} at {t[i]:+.2f} s")

json.dump({k:dict(ps=v['ps'].tolist(),
                  cl=[[int(a),int(b),float(c),float(d)] for a,b,c,d in v['cl']])
           for k,v in STORE.items()}, open(f'{DATA}/ts_stats.json','w'))

# ---------- individual curves per participant ----------
# -- the per-participant time-course figure (FigS_ts_individual.png) ---------------
# This figure is NOT part of the submitted manuscript; the submitted Figure S1 is the
# per-channel alpha and exponent figure produced by scripts/6_figures/figS2_channels.py.
# It is produced at the pixel size of an earlier draft version (4115 x 3306). An earlier
# version was cropped by bbox='tight' and came out 4700 x 4465, which had to be reduced and
# shrank the text. The size is fixed and the text enlarged to match. Final width 173 mm;
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '6_figures'))
import natstyle as _N
S1_DPI, S1_W, S1_H = _N.FIG_DPI, 4115, 3306      # W and H are the internal coordinate ratio
S1_TICK, S1_AXIS, S1_LET = _N.PT_TICK, _N.PT_AXIS, _N.PT_PANEL
S1_K = _N.pt_per_px(S1_W) / (72.0 / 400)         # line widths stay as they were
fig=plt.figure(figsize=_N.figsize(S1_W,S1_H),dpi=S1_DPI)
gs=gridspec.GridSpec(len(PIDS),3,figure=fig,wspace=.30,hspace=.55,
                     left=.115,right=.985,top=.930,bottom=.062)
for r,pid in enumerate(PIDS):
    for k,(key,lab) in enumerate(MEAS):
        ax=fig.add_subplot(gs[r,k])
        vc=np.array(T[pid]['fbC'][key],float); ve=np.array(T[pid]['fbE'][key],float)
        if len(vc)==n and not np.all(np.isnan(vc)):
            ax.plot(t,zbase(vc),color=EXP,lw=1.4*S1_K,label='unrewarded')
        if len(ve)==n and not np.all(np.isnan(ve)):
            ax.plot(t,zbase(ve),color=RED,lw=1.4*S1_K,label='rewarded')
        ax.axvline(0,color='#222',lw=.8,ls='--',zorder=1)
        ax.axhline(0,color='#ddd',lw=.5,zorder=1)
        ax.set_xlim(-PRE,t[-1])
        ax.tick_params(labelsize=S1_TICK)
        for s_ in ('top','right'): ax.spines[s_].set_visible(False)
        if k==0:
            ax.set_ylabel(f'{pid}',fontsize=S1_AXIS,fontweight='bold',
                          rotation=0,ha='right',va='center',labelpad=12)
        if r==0:
            ax.set_title(lab,fontsize=S1_AXIS,pad=4*S1_K)
            # The legend sits one line above the three column titles, so it does not cover them
            if k==2: ax.legend(loc='lower right',bbox_to_anchor=(1.0,1.42),
                               fontsize=S1_TICK,handlelength=1.0,ncol=2,
                               frameon=False,columnspacing=1.2,borderpad=0)
        if r==len(PIDS)-1: ax.set_xlabel('Time from feedback (s)',
                                         fontsize=S1_AXIS)
        else: ax.set_xticklabels([])
        # Styling kept from the draft version - the trial count goes **below** the participant
        # number in grey, and there is no figure title or subtitle (the caption carries the same content).
        nC=T[pid]['n_C']; nE=T[pid]['n_E']
        if k==0: ax.text(-.155,.22,f'{nC}/{nE}',transform=ax.transAxes,
                         fontsize=S1_TICK,color='#888',ha='right',va='center')
# bbox_inches=None falls back to rcParams('tight'). It is turned off explicitly.
rcParams['savefig.bbox']=None
plt.savefig(f'{OUT}/FigS_ts_individual.png',dpi=S1_DPI); plt.close()
rcParams['savefig.bbox']='tight'
print("\nindividual figures done")
