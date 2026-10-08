"""
spectro.py - the resting-state spectral pipeline

Produces the aperiodic exponent and the alpha peak amplitude from the resting EEG.
It is a direct implementation of the spectral-pipeline rewrite specification.

    python scripts/spectro.py                 main analysis
    python scripts/spectro.py --ica           with ICA (sensitivity)
    python scripts/spectro.py --sensitivity   the whole grid of combinations

Key decisions (specification section 2)

  ICA        not used in the main analysis. Resting state is eyes-closed, so blinks are
             few; there is no EOG channel and ICLabel cannot be used, which makes component
             judgement on 14 channels subjective. `--ica` is for the sensitivity run only.
  amplitude  200 uV for both rest and task. A different criterion per condition would turn
             a difference in rejection rate straight into a difference in the index.
  low-pass   none. The aperiodic exponent is sensitive to high-frequency attenuation, so a
             low-pass inflates it systematically. The hardware already limits to 0.2-45 Hz.
  notch      none (applied in hardware).
  fit        2-40 Hz, fixed mode. Above 40 Hz is the hardware roll-off, and below 1 Hz the
             eyes-closed spectrum bends easily.

Cluster indices are produced by **averaging the channel spectra first and fitting once**.
That differs in value from fitting per channel and averaging, so the two are never mixed.

On the marker files
  The specification named `*_markers.csv` as the input, but that file (the PsychoPy record)
  carries absolute epoch-ms times and cannot serve directly as an EDF sample index. What
  does hold the resting-window boundaries on the EDF time axis is Emotiv's
  `*_intervalMarker.csv`, and every other script here uses it. So does this one.
  The marker values are 10/11 for resting run 1 and 12/13 for run 2.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import argparse, csv, glob, json, shutil, warnings
import numpy as np
import mne
from scipy.signal import welch
from fooof import FOOOF

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

RAW = str(RAW); DATA = str(DATA); OUT = str(OUT)
CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8',
      'FC6', 'F4', 'F8', 'AF4']
FRONT = ['AF3', 'AF4', 'F3', 'F4']
POST = ['O1', 'O2']
# The parieto-occipital cluster of preregistration section 18.6. Not POST, the two occipital electrodes.
PO4 = ['O1', 'O2', 'P7', 'P8']
ICA_FRONT = ['AF3', 'AF4', 'F7', 'F8']

SFREQ = 128.0
EPOCH_S = 2.0            # 2 s non-overlapping -> 0.5 Hz resolution
SKIP_S = 10.0            # discard the first 10 s of each run (alpha transient after eye closure)
MIN_EPOCHS = 30          # drop a run if fewer epochs than this survive
R2_MIN = 0.85
AMP_DEFAULT = 200.0
FIT_DEFAULT = (2.0, 40.0)
ICA_SEED = 0
ICA_MAX_REMOVE = 2

PIDS = [f'{i:03d}' for i in range(1, 17)]


# ------------------------------------------------------------- I/O helpers

def find(pid, pat):
    return sorted(p for p in glob.glob(f'{RAW}/**/{pat}', recursive=True)
                  if os.path.basename(p).startswith(pid))


def rest_windows(marker_csv, tmax):
    """Get the (start, end) seconds of resting runs 1 and 2 from the markers."""
    M = [(int(r['marker_value']), float(r['latency']))
         for r in csv.DictReader(open(marker_csv, encoding='utf-8-sig'))]
    out = []
    for run, (sv, ev) in enumerate(((10, 11), (12, 13)), start=1):
        a = [t for v, t in M if v == sv]
        b = [t for v, t in M if v == ev]
        if not (a and b):
            continue
        s0 = a[-1]
        e0 = [t for t in b if t > s0]
        if not e0:
            continue
        out.append((run, s0, min(e0[0], tmax)))
    return out


def stim_times(marker_csv):
    M = [(int(r['marker_value']), float(r['latency']))
         for r in csv.DictReader(open(marker_csv, encoding='utf-8-sig'))]
    return sorted(t for v, t in M if v == 31)


# ------------------------------------------------------------- preprocessing

def load_raw(edf, detect_filter='qc'):
    """Read, pick channels, resample, filter (specification 3.1-3.2).

    Two copies are returned.

      analysis   0.5 Hz high-pass only. No low-pass, no notch (specification 3.2). The
                 aperiodic exponent is sensitive to high-frequency attenuation, so a
                 low-pass inflates it.  detection   1-45 Hz band-pass + 60 Hz notch.

    Why split them. Specification 3.3 asks for the bad-channel rates (P8 51.6, AF4 41.9,
    F7 and T8 6.5) to be reproduced, but judging under the filter of specification 3.2
    makes F7 12.9% and they do not reproduce. Splitting the conditions and checking, the
    only setting under which all four rates reproduce exactly was a 1-45 Hz band-pass
    plus notch with the first 10 s kept (F7: 12.9% as specified; 9.7% if only the filter
    changes; 12.9% if only the 10 s are dropped). The original filter was the band-pass.

    Bad-channel judgement is a judgement about electrode contact, so making it on a
    band-limited copy is defensible, and it compares directly against the published
    quality table. The spectra themselves come from the analysis copy, which follows
    specification 3.2. Pass `--detect-filter spec` to judge on the analysis filter too.
    """
    raw = mne.io.read_raw_edf(edf, preload=True)
    have = [c for c in CH if c in raw.ch_names]
    raw.pick(have)
    if abs(raw.info['sfreq'] - SFREQ) > 1e-6:
        raw.resample(SFREQ)
    ana = raw.copy().filter(0.5, None, method='fir', phase='zero', verbose=False)
    if detect_filter == 'spec':
        return ana, ana
    qc = raw.copy().filter(1.0, 45.0, method='fir', phase='zero', verbose=False)
    qc.notch_filter(60., verbose=False)
    return ana, qc


def fit_ica(raw, log_rows, pid):
    """Specification section 6 - ICA for the sensitivity run. Removes only components that are both frontally and low-frequency dominant."""
    ica_raw = raw.copy().filter(1.0, None, method='fir', phase='zero',
                                verbose=False)
    n_comp = len(raw.ch_names)                 # no average reference applied, so rank = n channels
    ica = mne.preprocessing.ICA(n_components=n_comp, method='infomax',
                                fit_params=dict(extended=True),
                                random_state=ICA_SEED, max_iter='auto')
    try:
        ica.fit(ica_raw)
    except Exception as e:
        log_rows.append(dict(pid=pid, n_components=n_comp, n_removed=0,
                             removed='', note=f'fit failed: {type(e).__name__}'))
        return raw
    mix = np.abs(ica.get_components())         # (channel, component)
    src = ica.get_sources(ica_raw).get_data()  # (component, sample)
    chs = raw.ch_names
    fi = [chs.index(c) for c in ICA_FRONT if c in chs]
    pi = [chs.index(c) for c in POST if c in chs]
    if not (fi and pi):
        log_rows.append(dict(pid=pid, n_components=n_comp, n_removed=0,
                             removed='', note='too few frontal/occipital channels'))
        return raw
    f, P = welch(src, fs=raw.info['sfreq'], nperseg=int(EPOCH_S * SFREQ), axis=-1)
    low = P[:, (f >= 0) & (f < 4)].sum(1) / np.maximum(P.sum(1), 1e-20)
    ratio = mix[fi].mean(0) / np.maximum(mix[pi].mean(0), 1e-12)
    cand = [k for k in range(n_comp) if ratio[k] >= 3.0 and low[k] >= 0.50]
    cand.sort(key=lambda k: -ratio[k])
    drop = cand[:ICA_MAX_REMOVE]
    log_rows.append(dict(pid=pid, n_components=n_comp, n_removed=len(drop),
                         removed=';'.join(f'IC{k}(ratio={ratio[k]:.1f},'
                                          f'low={low[k]:.2f})' for k in drop),
                         note=''))
    if not drop:
        return raw
    ica.exclude = drop
    out = raw.copy()
    ica.apply(out)
    return out


def epoch_run(D, fs, t0, t1, skip=SKIP_S):
    """Split a run into 2 s non-overlapping epochs, dropping the first skip seconds (specification 3.4).

    Bad-channel judgement uses skip=0: dropping the first 10 s changes the judgement and
    the rates of specification 5.1 stop reproducing (see the docstring of load_raw).
    """
    npts = int(EPOCH_S * fs)
    a0 = int((t0 + skip) * fs)
    a1 = int(t1 * fs)
    starts = [a0 + k * npts for k in range((a1 - a0) // npts)]
    ep = [D[:, a:a + npts] for a in starts if a >= 0 and a + npts <= D.shape[1]]
    return np.array(ep) if ep else np.empty((0, D.shape[0], npts))


def bad_channels(ptp):
    """Same rule as the earlier reconstruction (specification 3.3).

    A channel is bad if the median of its per-epoch peak-to-peak amplitude is more than
    three times, or less than 0.2 times, the median over all channels. Excluded, not interpolated.
    """
    med = np.median(ptp, 0)
    g = np.median(med)
    return [i for i in range(ptp.shape[1]) if med[i] > 3 * g or med[i] < 0.2 * g]


# ------------------------------------------------------------- fitting

def fit_spec(f, p, frange, mode):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.05,
               peak_threshold=2.0, aperiodic_mode=mode, verbose=False)
    try:
        fm.fit(f, p, list(frange))
    except Exception:
        return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < R2_MIN:
        return None
    return fm


def summarize(fm):
    pk = [q for q in fm.peak_params_ if 7.0 <= q[0] < 14.0]
    best = max(pk, key=lambda q: q[1]) if len(pk) else None
    return dict(exponent=float(fm.aperiodic_params_[-1]),
                offset=float(fm.aperiodic_params_[0]),
                r_squared=float(fm.r_squared_),
                peaks=[[float(x) for x in q] for q in fm.peak_params_],
                alpha_amp=(float(best[1]) if best is not None else None),
                alpha_cf=(float(best[0]) if best is not None else None),
                alpha_bw=(float(best[2]) if best is not None else None))


# ------------------------------------------------------------- per participant

def prepare(pid, use_ica, ica_log, detect_filter='qc'):
    """Read the EDF and build per-run epochs and the amplitude matrix (before the amplitude criterion)."""
    edfs = [p for p in find(pid, '*.edf') if not p.endswith('.md.edf')]
    mks = find(pid, '*_intervalMarker.csv')
    if not (edfs and mks):
        return None
    raw, rawqc = load_raw(edfs[0], detect_filter)
    if use_ica:
        raw = fit_ica(raw, ica_log, pid)
    fs = raw.info['sfreq']
    D = raw.get_data() * 1e6
    Dqc = rawqc.get_data() * 1e6
    chs = list(raw.ch_names)
    runs = []
    for run, t0, t1 in rest_windows(mks[0], raw.times[-1]):
        # detection: band-limited copy, first 10 s included (the condition that reproduces 5.1)
        epq = epoch_run(Dqc, fs, t0, t1, skip=0.0)
        if len(epq) == 0:
            continue
        bad = bad_channels(epq.max(2) - epq.min(2))
        # analysis: specification 3.2 filter, first 10 s dropped (specification 3.4)
        ep = epoch_run(D, fs, t0, t1)
        if len(ep) == 0:
            continue
        ptp = ep.max(2) - ep.min(2)                 # (epoch, channel)
        runs.append(dict(run=run, ep=ep, ptp=ptp, bad=bad, chs=chs, fs=fs,
                         n_epoch_total=len(ep)))
    task = None
    st = stim_times(mks[0])
    if len(st) >= 30:
        npts = int(EPOCH_S * fs)
        ept = np.array([D[:, int(t * fs):int(t * fs) + npts] for t in st
                        if int(t * fs) + npts <= D.shape[1]])
        if len(ept):
            task = dict(ep=ept, ptp=ept.max(2) - ept.min(2), chs=chs, fs=fs)
    return dict(runs=runs, task=task, chs=chs, fs=fs)


def psd_of(ep, keep, fs):
    """Epoch mean in linear power (specification 3.5). Hann window, 2 s."""
    f, P = welch(ep[:, keep], fs=fs, nperseg=int(EPOCH_S * fs),
                 window='hann', axis=-1)
    return f, P.mean(0)


def analyse(prep, amp, frange, mode, want_detail=False, min_epochs=MIN_EPOCHS):
    """Produce the per-run indices for a given amplitude criterion and fit setting.

    Scope of epoch rejection - this differs from the preregistration. Recorded, not changed.

    Registration section 28 reads "An epoch is rejected if the peak-to-peak amplitude **in any
    channel contributing to the cluster under analysis** exceeds a threshold"
    exceeds a threshold", that is, the decision is made **per cluster**. The implementation
    below judges across every surviving channel and is therefore stricter - one artefact in
    T8, for instance, also discards epochs of the parieto-occipital cluster.

    The difference in yield is large. Under common average, 150 uV and the 60 s criterion,
    parieto-occipital gives 6/15 participants under all-channel judgement and 12/15 under
    per-cluster judgement (feasibility.py and feasibility_prereg.py respectively). Lowering
    to 30 s brings all-channel judgement to 14/15 - that is why the minimum was lowered.

    The manuscript reports values from this implementation, so the default behaviour stands.
    The registration rule as written is implemented in scripts/3_hypotheses/spec_grid.py.
    """
    out = []
    for r in prep['runs']:
        keep = [i for i in range(len(r['chs'])) if i not in r['bad']]
        if not keep:
            continue
        ok = [k for k in range(len(r['ep'])) if r['ptp'][k][keep].max() <= amp]
        rec = dict(run=r['run'], n_total=r['n_epoch_total'], n_kept=len(ok),
                   bad=[r['chs'][i] for i in r['bad']],
                   chs=[r['chs'][i] for i in keep], dropped=len(ok) < min_epochs)
        if len(ok) < min_epochs:
            out.append(rec)
            continue
        f, P = psd_of(r['ep'][ok], keep, r['fs'])
        chs = rec['chs']
        for nm, want in (('front', FRONT), ('post', POST), ('po4', PO4)):
            ix = [chs.index(c) for c in want if c in chs]
            if not ix:
                continue
            # cluster: average the channel spectra first, then fit once
            fm = fit_spec(f, P[ix].mean(0), frange, mode)
            if fm is not None:
                rec[nm] = summarize(fm)
        if want_detail:
            tot = P[:, (f >= 1) & (f <= 45)].sum(1)
            rec['chan'] = {}
            for i, c in enumerate(chs):
                d = dict(alpha_rel=float(P[i][(f >= 8) & (f <= 12)].sum()
                                         / max(tot[i], 1e-20)))
                fm = fit_spec(f, P[i], frange, mode)
                if fm is not None:
                    d.update(exponent=summarize(fm)['exponent'],
                             alpha_amp=summarize(fm)['alpha_amp'],
                             r_squared=summarize(fm)['r_squared'])
                rec['chan'][c] = d
            rec['psd'] = dict(f=f.tolist(), P={c: P[i].tolist()
                                              for i, c in enumerate(chs)})
            fmF = None
            ix = [chs.index(c) for c in FRONT if c in chs]
            if ix:
                fmF = fit_spec(f, P[ix].mean(0), frange, mode)
            if fmF is not None:
                rec['flat'] = dict(f=fmF.freqs.tolist(),
                                   flat=(fmF.power_spectrum
                                         - fmF._ap_fit).tolist())
        out.append(rec)
    return out


def task_psd(prep, amp):
    t = prep['task']
    if t is None:
        return None
    bad = bad_channels(t['ptp'])
    keep = [i for i in range(len(t['chs'])) if i not in bad]
    ok = [k for k in range(len(t['ep'])) if t['ptp'][k][keep].max() <= amp]
    if len(ok) < MIN_EPOCHS:
        return None
    f, P = psd_of(t['ep'][ok], keep, t['fs'])
    chs = [t['chs'][i] for i in keep]
    return dict(f=f.tolist(), P={c: P[i].tolist() for i, c in enumerate(chs)},
                n_kept=len(ok), n_total=len(t['ep']))


# ------------------------------------------------------------- aggregation

def cluster_value(runs, nm, key):
    v = [r[nm][key] for r in runs
         if (not r['dropped']) and nm in r and r[nm].get(key) is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ica', action='store_true', help='apply ICA (sensitivity)')
    ap.add_argument('--sensitivity', action='store_true', help='the whole grid of combinations')
    ap.add_argument('--amp', type=float, default=AMP_DEFAULT)
    ap.add_argument('--min-epochs', type=int, default=MIN_EPOCHS,
                    dest='min_epochs',
                    help='minimum epochs per run (specification 3.4 says 30)')
    ap.add_argument('--detect-filter', choices=['qc', 'spec'], default='qc',
                    help='bad-channel detection filter. qc=1-45Hz+notch (default), '
                         'spec=the 0.5Hz high-pass of specification 3.2')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    # specification section 9 - back up the existing JSON
    old = os.path.join(DATA, '_old')
    os.makedirs(old, exist_ok=True)
    for n in ['spectro.json', 'flatspec.json', 'psd_all.json', 'psd_rest.json',
              'bandpow.json', 'chan_indices.json', 'chan_peak.json',
              'chan_exp_rest.json']:
        p = os.path.join(DATA, n)
        if os.path.exists(p) and not os.path.exists(os.path.join(old, n)):
            shutil.copy2(p, os.path.join(old, n))

    ica_log = []
    print('reading EDF ...')
    PREP = {}
    for pid in PIDS:
        p = prepare(pid, args.ica, ica_log, args.detect_filter)
        if p and p['runs']:
            PREP[pid] = p
            print(f'  {pid}: {len(p["runs"])} resting runs', flush=True)
    if ica_log:
        with open(os.path.join(OUT, 'ica_log.csv'), 'w', encoding='utf-8-sig',
                  newline='') as f:
            w = csv.DictWriter(f, fieldnames=['pid', 'n_components',
                                              'n_removed', 'removed', 'note'])
            w.writeheader(); w.writerows(ica_log)

    if args.sensitivity:
        run_sensitivity(PREP, args.min_epochs)
        return

    RES = {pid: analyse(PREP[pid], args.amp, FIT_DEFAULT, 'fixed',
                        want_detail=True, min_epochs=args.min_epochs)
           for pid in PREP}

    # -- 5.1 quality indices --------------------------------------
    n_run = sum(len(v) for v in RES.values())
    badc = {c: 0 for c in CH}
    for runs in RES.values():
        for r in runs:
            for c in r['bad']:
                badc[c] += 1
    r2 = [r[nm]['r_squared'] for runs in RES.values() for r in runs
          for nm in ('front', 'post', 'po4') if nm in r]
    kept = [r for runs in RES.values() for r in runs if not r['dropped']]

    print('\n' + '=' * 70)
    print(' 5.1 quality indices')
    print('=' * 70)
    TARGET = {'P8': 51.6, 'AF4': 41.9, 'F7': 6.5, 'T8': 6.5}
    print(f"  analysis runs       {n_run}          (target 31)")
    for c in CH:
        pct = badc[c] / n_run * 100 if n_run else 0
        tgt = TARGET.get(c, 0.0)
        mark = ' OK' if abs(pct - tgt) < 0.05 else (f'  <- target {tgt}%'
                                                    if (pct or tgt) else '')
        if pct or tgt:
            print(f"  {c:4s} bad {badc[c]:2d}/{n_run} = {pct:5.1f}%{mark}")
    print(f"  channels at 0%: "
          f"{', '.join(c for c in CH if badc[c] == 0)}")
    print(f"  median fit R2       {np.median(r2):.3f}      (target .98)")
    print(f"  runs passing the {args.min_epochs}-epoch criterion {len(kept)}/{n_run}")
    if args.min_epochs != 15:
        print(f"  Note. The manuscript values were produced with --min-epochs 15 (28/31 passed). "
              f"At {args.min_epochs} the sample differs from the manuscript.")
    nk = np.array([r['n_kept'] for runs in RES.values() for r in runs])
    print(f"  median surviving epochs {np.median(nk):.0f} (range {nk.min()}-{nk.max()})")
    if len(kept) < n_run * 0.8:
        print("")
        print(f"  [warning] the {args.min_epochs}-epoch threshold of specification 3.4 "
              f"discards {n_run - len(kept)} runs.")
        for thr in (10, 15, 20, 25, 30):
            print(f"         threshold {thr:2d} epochs -> passing "
                  f"{int((nk >= thr).sum()):2d}/{n_run} runs")

    # ── QC CSV ──────────────────────────────────────────────────
    sfx0 = '_ica' if args.ica else ''
    with open(os.path.join(OUT, f'spectro_qc{sfx0}.csv'), 'w',
              encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pid', 'run', 'n_epoch_total', 'n_epoch_kept', 'yield_pct',
                    'n_bad', 'bad_channels', 'dropped',
                    'front_exponent', 'front_alpha_amp', 'front_r2',
                    'post_exponent', 'post_alpha_amp', 'post_r2'])
        for pid in sorted(RES):
            for r in RES[pid]:
                g = lambda nm, k: (r[nm].get(k) if nm in r else None)
                w.writerow([pid, r['run'], r['n_total'], r['n_kept'],
                            round(r['n_kept'] / r['n_total'] * 100, 1),
                            len(r['bad']), ';'.join(r['bad']),
                            int(r['dropped']),
                            g('front', 'exponent'), g('front', 'alpha_amp'),
                            g('front', 'r_squared'),
                            g('post', 'exponent'), g('post', 'alpha_amp'),
                            g('post', 'r_squared')])

    # -- output JSON ----------------------------------------------
    # The schema follows exactly what the existing consumers (fig4_full.py, fig5_full.py,
    # resolve_B.py, subtype.py) expect, so the new pipeline's values can be swapped in
    # without touching the figure code.
    #
    #   psd_all[pid]   = {f, ch, rest1, rest2?, task?}      ch order = row order
    #   flatspec[pid]  = {f, rest:{flat}, task:{flat}}      frontal cluster, log residual
    #   chan_peak[pid][ch] = {pk: amplitude|None}
    #   bandpow[pid]   = {rest:{Delta..Gamma}, task:{...}}  frontal cluster relative power
    #
    # The band edges for bandpow are the ones written in section 2.4 of the manuscript
    # (alpha 8-13, beta 13-30). They differ from the PLV band definitions (alpha 8-12,
    # beta 14-30), and the manuscript treats the two as different indices.
    spectro = {}; flat = {}; psd_all = {}; psd_rest = {}
    bandpow = {}; chan_idx = {}; chan_peak = {}; chan_exp_rest = {}
    BANDS = {'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
             'Beta': (13, 30), 'Gamma': (30, 45)}

    def flat_of(f, P, chs):
        """The aperiodic-removed curve of the frontal cluster's mean spectrum."""
        ix = [chs.index(c) for c in FRONT if c in chs]
        if not ix:
            return None, None
        fm = fit_spec(f, P[ix].mean(0), FIT_DEFAULT, 'fixed')
        if fm is None:
            return None, None
        return fm.freqs, (fm.power_spectrum - fm._ap_fit)

    for pid, runs in RES.items():
        live = [r for r in runs if not r['dropped']]
        spectro[pid] = [{k: v for k, v in r.items()
                         if k in ('run', 'n_total', 'n_kept', 'bad', 'chs',
                                  'dropped', 'front', 'post', 'po4')} for r in runs]
        if not live:
            continue
        tp = task_psd(PREP[pid], args.amp)

        # use only channels that survive in every run and in the task for this participant
        sets = [set(r['chs']) for r in live] + ([set(tp['P'])] if tp else [])
        common = [c for c in CH if all(c in s for s in sets)]
        if not common:
            continue
        f = np.array(live[0]['psd']['f'])
        mat = lambda psd: np.array([psd['P'][c] for c in common])

        d = {'f': f.tolist(), 'ch': common}
        for r in live:
            d[f"rest{r['run']}"] = mat(r['psd']).tolist()
        if 'rest1' not in d:                     # only run 2 is valid
            d['rest1'] = d.pop('rest2')
        if tp:
            d['task'] = mat(tp).tolist()
        psd_all[pid] = d
        psd_rest[pid] = {'f': d['f'], 'ch': common, 'rest1': d['rest1']}

        # flatspec - rest is the run average, task is the task window
        Pr = np.mean([mat(r['psd']) for r in live], 0)
        ff, fr = flat_of(f, Pr, common)
        ft_, ftk = (flat_of(np.array(tp['f']), mat(tp), common)
                    if tp else (None, None))
        if ff is not None and ftk is not None:
            flat[pid] = {'f': ff.tolist(),
                         'rest': {'flat': fr.tolist()},
                         'task': {'flat': ftk.tolist()}}

        def rel(P, sel):
            ix = [common.index(c) for c in sel if c in common]
            if not ix:
                return None
            A = P[ix].mean(0)
            tot = A[(f >= 1) & (f <= 45)].sum()
            return {b: float(A[(f >= lo) & (f < hi)].sum() / tot)
                    for b, (lo, hi) in BANDS.items()}
        bp = {'rest': rel(Pr, FRONT), 'rest_post': rel(Pr, POST)}
        if tp:
            bp['task'] = rel(mat(tp), FRONT)
        if bp['rest'] and bp.get('task'):
            bandpow[pid] = bp

        acc = {}
        for r in live:
            for c, dd in r['chan'].items():
                acc.setdefault(c, []).append(dd)
        chan_idx[pid] = {c: dict(
            exponent=(float(np.mean([x['exponent'] for x in v if 'exponent' in x]))
                      if any('exponent' in x for x in v) else None),
            alpha_rel=float(np.mean([x['alpha_rel'] for x in v])))
            for c, v in acc.items()}
        chan_peak[pid] = {c: {'pk': (float(np.mean([x['alpha_amp'] for x in v
                                                    if x.get('alpha_amp') is not None]))
                                     if any(x.get('alpha_amp') is not None for x in v)
                                     else None)} for c, v in acc.items()}
        chan_exp_rest[pid] = {c: dd.get('exponent')
                              for c, dd in live[0]['chan'].items()}

    # flatspec consumers stack the per-participant arrays as they are, so the f grid must match
    if flat:
        n0 = len(next(iter(flat.values()))['f'])
        flat = {p: v for p, v in flat.items() if len(v['f']) == n0}

    # ICA is a sensitivity analysis. A suffix keeps it from overwriting the main-analysis output.
    sfx = '_ica' if args.ica else ''
    for nm, obj in [('spectro', spectro), ('flatspec', flat),
                    ('psd_all', psd_all), ('psd_rest', psd_rest),
                    ('bandpow', bandpow), ('chan_indices', chan_idx),
                    ('chan_peak', chan_peak), ('chan_exp_rest', chan_exp_rest)]:
        json.dump(obj, open(os.path.join(DATA, nm + sfx + '.json'), 'w'))
    qcname = f'spectro_qc{sfx}.csv'
    print(f'\nwritten: 8 JSON files in {DATA}{sfx and " (" + sfx + ")"}, {qcname} in {OUT}')

    lab = 'with ICA' if args.ica else 'main analysis'
    compare(RES, tag=f'{lab} - amplitude {args.amp:.0f} uV - minimum {args.min_epochs} epochs',
            path=f'spectro_compare{sfx}.csv')
    # once more at the threshold that preserves the sample - both are needed to judge the swap
    if args.min_epochs > 15:
        RES15 = {pid: analyse(PREP[pid], args.amp, FIT_DEFAULT, 'fixed',
                              min_epochs=15) for pid in PREP}
        compare(RES15, tag=f'{lab} - amplitude {args.amp:.0f} uV - minimum 15 epochs',
                path=f'spectro_compare_min15{sfx}.csv')


# ------------------------------------------------------------- 5.2 comparison table

def icc21(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    M = np.c_[a, b]; gm = M.mean()
    msr = 2 * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - gm) ** 2).sum()
    mse = ((M - M.mean(1)[:, None] - M.mean(0)[None, :] + gm) ** 2).sum() / (n - 1)
    return (msr - mse) / (msr + mse + 2 * (msc - mse) / n)


def behav():
    p = os.path.join(DATA, 'behav_metrics.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))


def pcorr(x, y, z):
    from scipy import stats as st
    rk = st.rankdata
    x, y, z = rk(x), rk(y), rk(z)
    rxy = np.corrcoef(x, y)[0, 1]; rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


SAMPLE9 = ['001', '002', '003', '004', '005', '006', '012', '014', '016']
OLD = {
    'frontal aperiodic exponent mean': 1.313, 'frontal aperiodic exponent SD': 0.347,
    'occipital aperiodic exponent mean': 1.118, 'occipital aperiodic exponent SD': 0.484,
    'ICC frontal exponent': .870, 'ICC frontal alpha': .921,
    'ICC occipital exponent': .983, 'ICC occipital alpha': .870,
    'exponent x switch-bias index rho': .733, 'exponent x lose-shift rho': .820,
    'exponent x perseverative errors rho': -.740, 'alpha x accuracy rho': .800,
    'alpha x trials-to-recover rho': -.817,
    'partial lose-shift x exponent | alpha': .808,
    'partial accuracy x alpha | exponent': .739,
}


def compare(RES, tag='main analysis', write=True, path='spectro_compare.csv'):
    from scipy import stats as st
    BM = behav()
    new = {}
    fe = {p: cluster_value(RES[p], 'front', 'exponent') for p in RES}
    fa = {p: cluster_value(RES[p], 'front', 'alpha_amp') for p in RES}
    pe = {p: cluster_value(RES[p], 'post', 'exponent') for p in RES}
    v = np.array([x for x in fe.values() if x is not None], float)
    new['frontal aperiodic exponent mean'] = float(v.mean()); new['frontal aperiodic exponent SD'] = float(v.std(ddof=1))
    v = np.array([x for x in pe.values() if x is not None], float)
    new['occipital aperiodic exponent mean'] = float(v.mean()); new['occipital aperiodic exponent SD'] = float(v.std(ddof=1))
    for nm, key, lab in [('front', 'exponent', 'ICC frontal exponent'),
                         ('front', 'alpha_amp', 'ICC frontal alpha'),
                         ('post', 'exponent', 'ICC occipital exponent'),
                         ('post', 'alpha_amp', 'ICC occipital alpha')]:
        A = []; B = []
        for p, runs in RES.items():
            live = {r['run']: r for r in runs if not r['dropped']}
            if 1 in live and 2 in live and nm in live[1] and nm in live[2]:
                a = live[1][nm].get(key); b = live[2][nm].get(key)
                if a is not None and b is not None:
                    A.append(a); B.append(b)
        new[lab] = float(icc21(A, B)) if len(A) >= 3 else None
        new[lab + ' n'] = len(A)
    if BM:
        S = [p for p in SAMPLE9 if fe.get(p) is not None and p in BM]
        E = [fe[p] for p in S]; A = [fa[p] for p in S]
        ok_a = all(x is not None for x in A)
        for lab, key, arr in [('exponent x switch-bias index rho', 'sbi', E),
                              ('exponent x lose-shift rho', 'lose_shift', E),
                              ('exponent x perseverative errors rho', 'n_persev', E),
                              ('alpha x accuracy rho', 'accuracy', A),
                              ('alpha x trials-to-recover rho', 'trials_to_recover', A)]:
            if arr is A and not ok_a:
                new[lab] = None; continue
            y = [BM[p][key] for p in S]
            new[lab] = float(st.spearmanr(arr, y).statistic)
        if ok_a and len(S) >= 4:
            ls = [BM[p]['lose_shift'] for p in S]; ac = [BM[p]['accuracy'] for p in S]
            new['partial lose-shift x exponent | alpha'] = float(pcorr(E, ls, A))
            new['partial accuracy x alpha | exponent'] = float(pcorr(A, ac, E))
        new['sample n'] = len(S)

    print('\n' + '=' * 78)
    print(f' 5.2 comparison table - {tag}')
    print('=' * 78)
    print(f"{'index':<40}{'original':>10}{'rebuilt':>10}{'difference':>12}")
    rows = []
    DASH = '-'.rjust(10)
    for k, oldv in OLD.items():
        nv = new.get(k)
        d = (nv - oldv) if (nv is not None) else None
        cn = f'{nv:>10.3f}' if nv is not None else DASH
        cd = f'{d:>+10.3f}' if d is not None else DASH
        print(f"{k:<40}{oldv:>10.3f}{cn}{cd}")
        rows.append({'index': k, 'original': oldv,
                     'rebuilt': (round(nv, 4) if nv is not None else ''),
                     'difference': (round(d, 4) if d is not None else '')})
    for k in ['sample n', 'ICC frontal exponent n', 'ICC occipital exponent n']:
        if k in new:
            print(f"{k:<40}{'':>10}{new[k]:>10}")
            rows.append({'index': k, 'original': '', 'rebuilt': new[k], 'difference': ''})
    if write:
        with open(os.path.join(OUT, path), 'w',
                  encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['index', 'original', 'rebuilt', 'difference'])
            w.writeheader(); w.writerows(rows)
        print(f'\nwritten: {OUT}/{path}')
    return new


# ------------------------------------------------------------- 5.3 sensitivity

def run_sensitivity(PREP, min_epochs=MIN_EPOCHS):
    from scipy import stats as st
    BM = behav()
    AMPS = [100.0, 150.0, 200.0, 250.0]
    MODES = ['fixed', 'knee']
    RANGES = [(2.0, 40.0), (1.0, 40.0), (2.0, 30.0)]
    # The minimum-epoch threshold is swept too. The 30 of specification 3.4 discards 21 of
    # the 31 runs and shrinks the sample from 9 participants to 6 - the largest single effect here.
    MINEPS = sorted({15, min_epochs})
    rows = []
    for me in MINEPS:
      for amp in AMPS:
        for mode in MODES:
            for fr in RANGES:
                RES = {p: analyse(PREP[p], amp, fr, mode, min_epochs=me)
                       for p in PREP}
                fe = {p: cluster_value(RES[p], 'front', 'exponent') for p in RES}
                fa = {p: cluster_value(RES[p], 'front', 'alpha_amp') for p in RES}
                S = [p for p in SAMPLE9 if fe.get(p) is not None and BM and p in BM]
                row = dict(amp_uV=amp, ica=int(bool(ICA_FLAG[0])), mode=mode,
                           fit_lo=fr[0], fit_hi=fr[1], min_epochs=me,
                           n=len(S),
                           n_runs_kept=sum(1 for runs in RES.values()
                                           for r in runs if not r['dropped']))
                if BM and len(S) >= 4:
                    E = [fe[p] for p in S]
                    row['rho_exp_loseshift'] = round(float(st.spearmanr(
                        E, [BM[p]['lose_shift'] for p in S]).statistic), 4)
                    row['rho_exp_sbi'] = round(float(st.spearmanr(
                        E, [BM[p]['sbi'] for p in S]).statistic), 4)
                    A = [fa[p] for p in S]
                    if all(x is not None for x in A):
                        row['rho_alpha_acc'] = round(float(st.spearmanr(
                            A, [BM[p]['accuracy'] for p in S]).statistic), 4)
                rows.append(row)
                print(f"  epochs>={me:2d} - amp {amp:.0f} - {mode:5s} - "
                      f"{fr[0]:.0f}-{fr[1]:.0f} Hz · n={row['n']} · "
                      f"rho(lose-shift)={row.get('rho_exp_loseshift','-')}",
                      flush=True)
    keys = ['amp_uV', 'ica', 'mode', 'fit_lo', 'fit_hi', 'min_epochs',
            'n', 'n_runs_kept',
            'rho_exp_loseshift', 'rho_exp_sbi', 'rho_alpha_acc']
    p = os.path.join(OUT, 'spectro_sensitivity.csv')
    exist = os.path.exists(p)
    with open(p, 'a' if exist else 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exist:
            w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    print(f'\nwritten: {p}  ({len(rows)} combinations)')


ICA_FLAG = [False]

if __name__ == '__main__':
    ICA_FLAG[0] = '--ica' in sys.argv
    main()
