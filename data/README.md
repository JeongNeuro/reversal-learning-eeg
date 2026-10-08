# data — what to put here

Everything in this folder is listed in `.gitignore` and is never committed.
It holds participant data, so **do not push it to a public repository.**

## 1. Required files

| File | Contents | Scripts that need it |
|---|---|---|
| `raw/*_behav.csv` | trial-level behavioural record | `behav_metrics` and almost everything downstream |
| `raw/*.edf` | raw EEG | `spectro`, `tfr`, `timeseries`, `connectivity`, `w1`, `phases`, `timescale`, `timescale_checks` |
| `raw/*_intervalMarker.csv` | event markers | `w1`, `phases`, `tfr` |

With those three, `run_all.py` runs end to end. The sample definition and the
behavioural and EEG indices are derived from `spectro.json` and `behav_metrics.json`
by `cohort.py` at the repository root. `behav_metrics.py` needs only the behavioural
CSV and runs without the EDF files.

## 1.1 Optional files

| File | Contents | What happens without it |
|---|---|---|
| `demographics.csv` | columns `id, age, sex, disability, psychotropic_med` | only the age-confound section of `robustness`, the confound section of `subtype_validate`, and the demographic columns of `persev_direct` are skipped. Everything else runs |
| `questionnaire.csv` | columns `id, i01 … i20` (RBQ-2A item responses, 1–3) | `rbq.py` exits without doing anything |
| `ASD_study_data.xlsx` | the team's internal summary workbook | `excel_recompute` skips its comparison and exits. No other script reads this file |

The workbook's own filename and sheet names are not in English in the original. Point
`ASD_XLSX` at the file to use another name, and set `ASD_XLSX_SHEETS` to the real sheet
names, comma separated, in this order: participants, behaviour, eeg, eeg_quality,
questionnaire, analysis, sensitivity. Without the override the sheets are taken by
position in that order.

The `disability` column of `demographics.csv` is free text. `subtype_validate` and
`persev_direct` classify it by case-insensitive substring, with English defaults. If the
column is written in another language, set `ASD_DISABILITY_AUTISM`, `ASD_DISABILITY_DOWN`,
`ASD_DISABILITY_MULTIPLE`, `ASD_DISABILITY_DEVELOPMENTAL`, `ASD_DISABILITY_INTELLECTUAL`,
`ASD_DISABILITY_GRADE1` or `ASD_DISABILITY_GRADE3` to comma-separated substrings.

**Both of those first two are participant data.** `.gitignore` blocks the whole of
`data/`, so they will not be committed by accident, but do not place them in a public
repository.

## 2. Filename convention inside raw/

Use exactly these names inside `raw/`. Subfolders are fine — the scripts search
recursively.

The `ASD_` prefix, and the `ASD_*` environment variables that follow it, are the
file-naming convention used during recording. They are not a statement about the
sample: the study recruited adults with registered developmental disabilities, most
of whom have an intellectual disability. The prefix is kept because it is what the
recorded files on disk are actually called, and the scripts parse that pattern.

```
ASD_<8-digit date>_<3-digit participant>_behav.csv      trial-level behaviour  (required)
ASD_<8-digit date>_<3-digit participant>_markers.csv    event markers
ASD_<8-digit date>_<3-digit participant>_meta.json      session metadata
ASD_<8-digit date>_<3-digit participant>.edf            raw EEG                (needed for preprocessing)
```

For example:

```
raw/ASD_20260811_001_behav.csv
raw/ASD_20260811_001_markers.csv
raw/ASD_20260811_001_meta.json
raw/ASD_20260811_001.edf
```

Participant numbers run from 001 to 016, and behavioural files exist for the 12
participants who performed the task (001–009, 012, 014, 016).

**Careful.** The scripts read the second-to-last segment of the filename as the
participant number (`ASD_20260818_008_behav.csv` → `008`). If you add another number
in the middle to mark a rerun or a partial session, as in
`ASD_20260818_008_1_behav.csv`, the participant number is misread as `1`. Rename such
files or keep them outside `raw/`.

## 3. Intermediate files (JSON)

These are produced from the EDF files by the preprocessing scripts. If you already
have them, drop them into `data/` and preprocessing can be skipped.

| File | Produced by | Used for |
|---|---|---|
| `spectro.json` | `spectro.py` | Figure 4a |
| `flatspec.json` | `spectro.py` | periodic components by band, Results 3.4 |
| `psd_all.json`, `psd_rest.json` | `spectro.py` | individual alpha frequency, per-channel exponent |
| `bandpow.json` | `spectro.py` | Figure 5a |
| `chan_indices.json`, `chan_peak.json`, `chan_exp_rest.json` | `spectro.py` | topographies in Figure 4 and Figure S1 |
| `conn.json` | `connectivity.py` | Figure 5b–e and the network metrics |
| `network_recomputed.json` | `network_recompute.py` | network values, Results 3.4 |
| `tfr.json` | `tfr.py` | correct-versus-error comparisons |
| `w1.json` | `w1.py` | within-participant hypotheses |
| `phases.json` | `phases.py` | comparisons by learning phase |
| `persev.json`, `trajectory.json` | `persev_direct.py`, `trajectory.py` | Figure 3b |
| `timeseries.json`, `ts_stats.json`, `timescale.json`, `timescale_checks.json` | scripts of the same name | supplementary material |
| `subtype.json`, `subtype_valid.json`, `hmm_valid.json`, `rlfit.json` | scripts of the same name | supplementary material (not used in the main text) |
| `peaks.json`, `peak_profile.json` | `peaks_gamma.py` | gamma temporal dominance |

**`peaks_gamma.py` is not yet in this repository.** If the code that produced
`peaks.json` and `peak_profile.json` turns up, add it to `scripts/`.
