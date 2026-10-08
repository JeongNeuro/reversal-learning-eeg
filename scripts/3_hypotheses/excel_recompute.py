"""
excel_recompute.py - recompute every manuscript number from the internal workbook alone
following the codebook definitions exactly.

This is the only script that takes the workbook as input. Its purpose is to check that the
team's internal summary file and the pipeline outputs give the same values. The workbook is
participant data and is not in the public repository, so if the file is absent this script
does nothing and exits (run_all.py does not stop here). Every other script in the repository
runs from the pipeline outputs alone. Point ASD_XLSX at the file to use it.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import numpy as np, math, warnings, json
from scipy import stats
warnings.filterwarnings("ignore")

if not os.path.exists(str(XLSX)):
    print(f"{os.path.basename(str(XLSX))} not found - skipping the workbook comparison.")
    print("This comparison runs only when the team's internal summary file is present.")
    raise SystemExit(0)

from openpyxl import load_workbook
wb=load_workbook(str(XLSX),data_only=True)
# The sheets of the internal workbook are named in Korean, so they are resolved by position
# in the order of SHEET_ORDER below and this file needs no Korean itself. An exact English
# sheet name is used when one exists. Override with ASD_XLSX_SHEETS: a comma-separated list
# of the real sheet names, in the same order.
SHEET_ORDER = ['participants', 'behaviour', 'eeg', 'eeg_quality',
               'questionnaire', 'analysis', 'sensitivity']
_SHEET_OVERRIDE = [s.strip() for s in os.environ.get('ASD_XLSX_SHEETS', '').split(',') if s.strip()]


def sheet(role):
    _i = SHEET_ORDER.index(role)
    if _SHEET_OVERRIDE:
        ws = wb[_SHEET_OVERRIDE[_i]]
    elif role in wb.sheetnames:
        ws = wb[role]
    else:
        ws = wb[wb.sheetnames[_i]]
    h=[ws.cell(row=1,column=i).value for i in range(1,ws.max_column+1)]
    out=[]
    for r in range(2,ws.max_row+1):
        d={h[i]:ws.cell(row=r,column=i+1).value for i in range(len(h)) if h[i]}
        if any(v is not None for v in d.values()): out.append(d)
    return out
PT=sheet('participants'); BH=sheet('behaviour'); EG=sheet('eeg')
QC=sheet('eeg_quality'); SV=sheet('questionnaire'); AN=sheet('analysis'); SN=sheet('sensitivity')

def boot(x,y,n=10000,seed=0):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        s=rng.choice(len(x),len(x),replace=True)
        xs=[x[i] for i in s]; ys=[y[i] for i in s]
        if len(set(xs))<3 or len(set(ys))<3: continue
        r=stats.spearmanr(xs,ys).statistic
        if not math.isnan(r): v.append(r)
    return (np.percentile(v,2.5),np.percentile(v,97.5)) if v else (np.nan,np.nan)
def perm(x,y,n=20000,seed=1):
    rng=np.random.default_rng(seed); o=abs(stats.spearmanr(x,y).statistic); c=0
    for _ in range(n):
        if abs(stats.spearmanr(x,rng.permutation(y)).statistic)>=o: c+=1
    return (c+1)/(n+1)
def L(t): print("\n"+"="*76+f"\n {t}\n"+"="*76)

# ---------- 1. participants ----------
L("1. participants")
print(f"  {len(PT)} in total")
st={}
for p in PT: st[p['status']]=st.get(p['status'],0)+1
for k,v in st.items(): print(f"    {k}: {v}")
ages=[p['age'] for p in PT if p.get('age')]
print(f"  age mean {np.mean(ages):.1f} (SD {np.std(ages,ddof=1):.1f}) - range {min(ages)}-{max(ages)}, n={len(ages)}")
sx={}
for p in PT:
    if p.get('sex'): sx[p['sex']]=sx.get(p['sex'],0)+1
print(f"  sex {sx}")
med=[p['psychotropic_med'] for p in PT if p.get('psychotropic_med') is not None]
print(f"  taking psychotropic medication {sum(med)} / not taking {len(med)-sum(med)}")
dis={}
for p in PT:
    if p.get('disability'): dis[p['disability']]=dis.get(p['disability'],0)+1
print(f"  registered disability type: {dis}")

# ---------- 2. behaviour ----------
L("2. behavioural indices")
print(f"  {len(BH)} with behavioural data")
def desc(key,lab,mul=1):
    v=[b[key]*mul for b in BH if b.get(key) is not None]
    print(f"    {lab:>22}: mean {np.mean(v):.3f} (SD {np.std(v,ddof=1):.3f}) - "
          f"range {min(v):.3f}-{max(v):.3f}")
for k,l in [('sbi','switch-bias index'),('n_persev','perseverative errors'),('n_regress','regressive errors'),
            ('lose_shift','lose-shift'),('win_stay','win-stay'),('accuracy','accuracy'),
            ('switch_rate','switch rate'),('trials_to_recover','trials to recover')]:
    desc(k,l)
pv=[b['n_persev'] for b in BH]; rg=[b['n_regress'] for b in BH]
r=stats.spearmanr(pv,rg)
print(f"  perseverative x regressive: rho = {r.statistic:+.3f}, P = {r.pvalue:.4f}")
sb=[b['sbi'] for b in BH]; sw=[b['switch_rate'] for b in BH]
print(f"  SBI x switch rate: r = {stats.pearsonr(sb,sw)[0]:+.3f}")
npd=sum(1 for b in BH if b['sbi']<0)
print(f"  perseverative-dominant {npd} - regressive-dominant {len(BH)-npd}")
for b in BH:
    if b['id'] in ('004','003'):
        print(f"  {b['id']}: perseverative {b['n_persev']} regressive {b['n_regress']} "
              f"sum {b['n_persev']+b['n_regress']} - win-stay {b['win_stay']:.3f} "
              f"· lose-shift {b['lose_shift']:.3f}")

# ---------- 3. EEG x behaviour ----------
L("3. resting EEG and behaviour (analysis sheet, exclusions applied)")
# The front_exp / front_alpha of the workbook are the old pipeline's output. Where
# spectro.json exists, those values are substituted. As in fig4_full.py, the sample
# composition is not changed; only the values of participants who already had one are updated.
def _spec_mean(pid, clus, key):
    v = []
    for r in _SPEC.get(pid, []):
        if r.get('dropped') or not r.get(clus):
            continue
        x = r[clus].get(key)
        if x is not None:
            v.append(x)
    return float(np.mean(v)) if v else None

try:
    _SPEC = json.load(open(f'{DATA}/spectro.json'))
except FileNotFoundError:
    _SPEC = {}
    print("  Note. spectro.json is absent, so the workbook's old pipeline values are used as they are.")
if _SPEC:
    for a in AN:
        if a.get('front_exp') is None or a.get('front_alpha') is None:
            continue
        _e = _spec_mean(a['id'], 'front', 'exponent')
        _a = _spec_mean(a['id'], 'front', 'alpha_amp')
        if _e is not None: a['front_exp'] = _e
        if _a is not None: a['front_alpha'] = _a

S=[a for a in AN if not a.get('excl_winstay') and a.get('front_exp') is not None
   and a.get('lose_shift') is not None]
print(f"  n = {len(S)}: {', '.join(a['id'] for a in S)}"
      f"{'  (spectro.json values)' if _SPEC else '  (old workbook values)'}\n")
E=[a['front_exp'] for a in S]; A=[a['front_alpha'] for a in S]
print(f"{'behavioural index':>24}{'exponent rho':>14}{'95% CI':>20}{'alpha rho':>12}{'95% CI':>20}")
for k,l in [('lose_shift','lose-shift'),('n_persev','perseverative errors'),('sbi','switch-bias index'),
            ('switch_rate','switch rate'),('trials_to_recover','trials to recover'),
            ('accuracy','accuracy'),('n_regress','regressive errors')]:
    y=[a.get(k) for a in S]
    if any(v is None for v in y): continue
    r1=stats.spearmanr(E,y).statistic; c1=boot(E,y)
    r2=stats.spearmanr(A,y).statistic; c2=boot(A,y)
    m1='*' if (c1[0]>0 or c1[1]<0) else ' '
    m2='*' if (c2[0]>0 or c2[1]<0) else ' '
    print(f"{l:>20}{r1:>+9.3f}{m1}  [{c1[0]:>+.2f}, {c1[1]:>+.2f}]"
          f"{r2:>+9.3f}{m2}  [{c2[0]:>+.2f}, {c2[1]:>+.2f}]")
print(f"\n  exponent x alpha: rho = {stats.spearmanr(E,A).statistic:+.3f}")

# partial correlations
L("4. partial correlations (rank based)")
def pcorr(x,y,z):
    rk=stats.rankdata; x,y,z=rk(x),rk(y),rk(z)
    rxy=stats.pearsonr(x,y)[0]; rxz=stats.pearsonr(x,z)[0]; ryz=stats.pearsonr(y,z)[0]
    r=(rxy-rxz*ryz)/math.sqrt((1-rxz**2)*(1-ryz**2)); n=len(x); df=n-3
    t=r*math.sqrt(df/(1-r**2)); return r,2*(1-stats.t.cdf(abs(t),df)),df
def permp(x,y,z,n=20000,seed=1):
    rng=np.random.default_rng(seed); o=abs(pcorr(x,y,z)[0]); c=0
    for _ in range(n):
        try:
            if abs(pcorr(x,rng.permutation(y),z)[0])>=o: c+=1
        except Exception: pass
    return (c+1)/(n+1)
LS=[a['lose_shift'] for a in S]; AC=[a['accuracy'] for a in S]
for lab,x,y,z in [('lose-shift x exponent | alpha',E,LS,A),('lose-shift x alpha | exponent',A,LS,E),
                  ('accuracy x exponent | alpha',E,AC,A),('accuracy x alpha | exponent',A,AC,E)]:
    r,p,df=pcorr(x,y,z); pp=permp(x,y,z)
    print(f"  {lab:>30}: r = {r:+.3f}  df = {df}  t-test P = {p:.4f}  permutation P = {pp:.4f}")

# Steiger
L("5. Steiger - difference of two correlations")
def steiger(x1,x2,y,n):
    rk=stats.rankdata
    r12=stats.pearsonr(rk(x1),rk(y))[0]; r13=stats.pearsonr(rk(x2),rk(y))[0]
    r23=stats.pearsonr(rk(x1),rk(x2))[0]
    z12=math.atanh(np.clip(r12,-.9999,.9999)); z13=math.atanh(np.clip(r13,-.9999,.9999))
    rm2=(r12**2+r13**2)/2; f=min((1-r23)/(2*(1-rm2)),1); hh=(1-f*rm2)/(1-rm2)
    z=(z12-z13)*math.sqrt((n-3)/(2*(1-r23)*hh))
    return z,2*(1-stats.norm.cdf(abs(z)))
for lab,Y in [('lose-shift',LS),('accuracy',AC),('SBI',[a['sbi'] for a in S]),
              ('perseverative errors',[a['n_persev'] for a in S])]:
    z,p=steiger(E,A,Y,len(S))
    print(f"  {lab:>12}: z = {z:+.2f}, P = {p:.4f}")

# ---------- 6. sensitivity ----------
L("6. sensitivity (the 144 rows of the sensitivity sheet)")
v=[s['rho_sbi'] for s in SN if isinstance(s.get('rho_sbi'),(int,float))]
print(f"  {len(v)} combinations - median {np.median(v):+.3f} - range {min(v):+.3f} to {max(v):+.3f}")
print(f"  {sum(1 for x in v if x<0)} combinations negative")
print("  Note. This sheet is reject_uv x bad_ch_excl x fit_range x r2_min x run_combine")
print("       = 144 combinations. The grid of preregistration 32.4 is reference x threshold x")
print("       fit_range x aperiodic_mode x cluster = 144, which has different factors, and the")
print("       96 combinations of spectro.py --sensitivity differ again. None of the three matches")
print("       the registered grid - the registered grid has reference and cluster as factors and")
print("       neither appears in the other two.")
vp=[s['rho_persev'] for s in SN if isinstance(s.get('rho_persev'),(int,float))]
print(f"  perseverative correlation: median {np.median(vp):+.3f} - range {min(vp):+.3f} to {max(vp):+.3f}")

# ---------- 7. quality ----------
L("7. EEG quality (eeg_quality sheet)")
print(f"  {len(QC)} runs in total")
CH=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
cnt={c:0 for c in CH}
for q in QC:
    bc=q.get('bad_channels')
    if bc and str(bc).strip() not in ('','None','-'):
        for c in str(bc).replace(' ','').split(','):
            if c in cnt: cnt[c]+=1
print(f"  {'channel':>9}{'bad':>6}{'rate':>9}")
for c,n in sorted(cnt.items(),key=lambda x:-x[1]):
    if n: print(f"{c:>6}{n:>6}{n/len(QC)*100:>8.1f}%")
r2f=[q['r2_front'] for q in QC if isinstance(q.get('r2_front'),(int,float))]
r2p=[q['r2_post'] for q in QC if isinstance(q.get('r2_post'),(int,float))]
print(f"  median R2 frontal {np.median(r2f):.3f} (n={len(r2f)}) - occipital {np.median(r2p):.3f} (n={len(r2p)})")
kf=[q['kept_front']/q['total_front'] for q in QC
    if q.get('total_front') and q.get('kept_front') is not None]
print(f"  median frontal epoch yield {np.median(kf)*100:.1f}%")

# ══════════ 8. ICC ══════════
L("8. run-to-run ICC(2,1), absolute agreement")
def icc21(a,b):
    a=np.array(a,float); b=np.array(b,float); n=len(a)
    M=np.column_stack([a,b]); gm=M.mean()
    msr=2*((M.mean(1)-gm)**2).sum()/(n-1)
    msc=n*((M.mean(0)-gm)**2).sum()/1
    mse=((M-M.mean(1)[:,None]-M.mean(0)[None,:]+gm)**2).sum()/(n-1)
    return (msr-mse)/(msr+mse+2*(msc-mse)/n)
def _spec_runs(pid, clus, key):
    """The surviving run values of spectro.json, in order."""
    out = []
    for r in _SPEC.get(pid, []):
        if r.get('dropped') or not r.get(clus):
            continue
        x = r[clus].get(key)
        if x is not None:
            out.append(x)
    return out

for k1,k2,lab,_cl,_ky in [
        ('front_exp_r1','front_exp_r2','frontal aperiodic exponent','front','exponent'),
        ('front_alpha_r1','front_alpha_r2','frontal alpha','front','alpha_amp'),
        ('post_exp_r1','post_exp_r2','occipital aperiodic exponent','post','exponent'),
        ('post_alpha_r1','post_alpha_r2','occipital alpha','post','alpha_amp')]:
    if _SPEC:
        pr=[(_v[0],_v[1]) for e in EG
            for _v in [_spec_runs(e['id'],_cl,_ky)] if len(_v)>=2]
    else:
        pr=[(e[k1],e[k2]) for e in EG
            if isinstance(e.get(k1),(int,float)) and isinstance(e.get(k2),(int,float))]
    if len(pr)<4: continue
    a=[p[0] for p in pr]; b=[p[1] for p in pr]
    print(f"  {lab:>18}: ICC = {icc21(a,b):.3f}  (n={len(pr)})")

# ---------- 9. sample characteristics ----------
L("9. sample characteristics")
# Here too the workbook holds the old pipeline. Where spectro.json exists, its values are used.
def _spec_all(clus, key):
    out = []
    for pid in sorted(_SPEC):
        v = [r[clus][key] for r in _SPEC[pid]
             if not r.get('dropped') and r.get(clus) and r[clus].get(key) is not None]
        if v:
            out.append(float(np.mean(v)))
    return out

_src = 'spectro.json' if _SPEC else 'old workbook values'
fe = _spec_all('front','exponent') if _SPEC else \
     [e['front_exp'] for e in EG if isinstance(e.get('front_exp'),(int,float))]
print(f"  frontal aperiodic exponent: mean {np.mean(fe):.3f} (SD {np.std(fe,ddof=1):.3f}), n={len(fe)}  ({_src})")
pe = _spec_all('post','exponent') if _SPEC else \
     [e['post_exp'] for e in EG if isinstance(e.get('post_exp'),(int,float))]
print(f"  occipital aperiodic exponent: mean {np.mean(pe):.3f} (SD {np.std(pe,ddof=1):.3f}), n={len(pe)}  ({_src})")
ac=[b['accuracy'] for b in BH]
print(f"  task accuracy: mean {np.mean(ac):.3f} - max {max(ac):.3f} - above .70 {sum(1 for x in ac if x>.70)}")

# ---------- 10. questionnaire ----------
L("10. RBQ-2A")
S2=[a for a in AN if a.get('rbq_is') is not None and a.get('sbi') is not None]
print(f"  {len(S2)} participants with both RBQ and behaviour")
for k,l in [('rbq_is','insistence on sameness'),('rbq_total','total')]:
    y=[a[k] for a in S2]; x=[a['sbi'] for a in S2]
    r=stats.spearmanr(x,y); c=boot(x,y)
    print(f"    {l:>10} × SBI: ρ = {r.statistic:+.3f}  [{c[0]:+.2f}, {c[1]:+.2f}]")
isv=[s['rbq_is'] for s in SV if isinstance(s.get('rbq_is'),(int,float))]
print(f"  insistence-on-sameness score: n={len(isv)} - range {min(isv):.3f}-{max(isv):.3f} - "
      f"{sum(1 for x in isv if x<=1.0)} at the floor")
items=[f'i{i:02d}' for i in [11,12,13,14,15,16,17,19]]
rows=[[s[c] for c in items] for s in SV if all(isinstance(s.get(c),(int,float)) for c in items)]
if len(rows)>3:
    M=np.array(rows,float); k=M.shape[1]
    a=k/(k-1)*(1-M.var(0,ddof=1).sum()/M.sum(1).var(ddof=1))
    print(f"  Cronbach alpha (items 11-17, 19) = {a:.3f}, n={len(rows)}")
