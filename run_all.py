"""run_all.py — rerun the analysis, stage by stage.

    python run_all.py              everything
    python run_all.py --check      preregistered tests and recomputation only (stages 1-3)
    python run_all.py --list       print what runs, and in what order, without running it
    python run_all.py --skip-slow  skip the stages that re-read the raw EDF files

The folder names under scripts/ are the execution order.

  1_preprocess   read the raw EDF and CSV files and write intermediate .json. Slow.
  2_metrics      reshape that .json into the form the paper uses.
  3_hypotheses   test the preregistered hypotheses.
  4_validation   shake the results to see whether they hold. Several end in "unusable".
  5_exploratory  analyses left out of the main text or moved to the supplement.
  6_figures      the six figures in the paper. Figures 4 and 5 compute some of the
                 statistics they need themselves, and the same values are also
                 produced in 3_hypotheses. See the warning below.

The code that typesets the manuscript is not in this repository. It is not analysis
code, and it could drift away from the published version.

Warning. The partial correlations, permutation tests and Steiger tests of section 3.3
are computed in two places - 3_hypotheses/pvalues.py and panels i and j of
6_figures/fig4_full.py. The definitions and the random seed currently match, so the
values agree. If you change one, check the other. That kind of drift has happened
twice in this repository.

Note the --min-epochs 15 passed to spectro.py. The script defaults to 30 (= 60 s),
the criterion set in section 28.7 of the preregistration, but the manuscript values
were computed at 15 (= 30 s). Running with the default leaves 10 of 31 runs instead
of 28 of 31, which shrinks the analysis sample well below 9 participants.

Stage 1 stops if the raw data are missing. See data/README.md.
"""
import subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
S = ROOT / 'scripts'

STAGES = [
    ('1_preprocess',   ['clock_offset.py',
                    ('spectro.py', ['--min-epochs', '15']),
                    'tfr.py', 'timeseries.py',
                    'connectivity.py', 'behav_metrics.py']),
    ('2_metrics',     ['network_recompute.py', 'prereg_clusters.py']),
    # w2_prereg_trials.py re-reads the raw EDF and is slow. Use --skip-slow to skip
    # it when W2_prereg_trials.csv already exists.
    # w1_prereg.py and w2_boot_p.py produce values that pvalues.py reads, so the
    # order here matters.
    # w1_prereg_windows.py rebuilds the windows from the raw data (slow). It writes
    # W1_windows_rebuild.csv rather than overwriting the W1_prereg_windows.csv that
    # the manuscript used.
    ('3_hypotheses', ['w1.py', ('w1_prereg_windows.py', [], 'slow'), 'w1_prereg.py',
                    'w2_theta.py',
                    ('w2_prereg_trials.py', [], 'slow'), 'w2_prereg_fit.py',
                    'w2_boot_p.py',
                    'spec_grid.py', 'band_window_grid.py', 'band_window_grid_prereg.py', 'pvalues.py',
                    'topo_cluster_perm.py',
                    'robustness.py',
                    'a7_mantel.py', 'excel_recompute.py']),
    ('4_validation',     ['rlmodel.py', 'hmm_validate.py', 'subtype.py',
                    'subtype_validate.py', 'ts_stats.py', 'feasibility.py',
                    'feasibility_prereg.py']),
    ('5_exploratory',     ['timescale.py', 'timescale_checks.py', 'phases.py',
                    'resolve_B.py', 'persev_direct.py', 'trajectory.py',
                    'rbq.py', 'prestim_alpha.py', 'prestim_alpha_fit.py']),
    # The supplementary time-course figure is produced by ts_stats.py in
    # 4_validation, so it is not repeated here.
    # The six figures of the submitted manuscript. natstyle.py is the module that
    # holds the specification, so it is not listed. fig4_scatter.py produces panels
    # d-h of Figure 4 on its own, and fig4_full.py imports its sizing and
    # point-labelling rules from it.
    ('6_figures', ['fig1_task.py', 'figM_flow.py', 'fig3_errors.py',
                    'figS2_channels.py', 'fig4_scatter.py', 'fig4_full.py',
                    'fig5_full.py']),
]

CHECK_ONLY = ('1_preprocess', '2_metrics', '3_hypotheses')


def parse(item):
    """An item is one of 'name', ('name', args) or ('name', args, 'slow')."""
    if isinstance(item, tuple):
        return (item + ([], None))[:3] if len(item) < 3 else item
    return (item, [], None)


def run(stage, item):
    name, args, tag = parse(item)
    path = S / stage / name
    print(f'\n[{stage}/{name}]' + (' ' + ' '.join(args) if args else ''))
    r = subprocess.run([sys.executable, str(path), *args])
    if r.returncode:
        raise SystemExit(f'{stage}/{name} failed')


if __name__ == '__main__':
    stages = STAGES
    if '--check' in sys.argv:
        stages = [(s, f) for s, f in STAGES if s in CHECK_ONLY]
    if '--list' in sys.argv:
        for s, fs in stages:
            print(s)
            for f in fs:
                nm, ar, tag = parse(f)
                print('   ', nm, ' '.join(ar), '(slow)' if tag == 'slow' else '')
        raise SystemExit
    skip_slow = '--skip-slow' in sys.argv
    for s, fs in stages:
        for f in fs:
            if skip_slow and parse(f)[2] == 'slow':
                print(f'\n[{s}/{parse(f)[0]}] skipped (--skip-slow)')
                continue
            run(s, f)
    print('\nDone - figures and tables are in outputs/.')
