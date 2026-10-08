# task — the presentation program

The PsychoPy scripts that were actually run during the sessions. This code produces
the behavioural CSV, the marker file and the session metadata found in `data/raw/`.

| File | What it does |
|---|---|
| `prl_task.py` | the probabilistic reversal-learning task itself: two practice stages, then 120 trials |
| `lsl_client.py` | sends markers to the EEG system over LSL. **All 18 sessions were recorded through this path** |
| `cortex_client.py` | connects through the Emotiv Cortex API. An alternative path that was not used in practice |
| `test_lsl.py` | checks the LSL connection |
| `make_schedule.py` | regenerates the seed-0 trial schedule as a CSV (see "Trial schedule" below) |
| `seed0_schedule.csv` | that output, for all 120 trials |
| `requirements.txt` | packages needed only to run the task; not needed for analysis |

The marker path is selected by `BACKEND` in `prl_task.py`. The value used is recorded
in the `backend` field of each session's `meta.json`.

## Error-type classification

Perseverative and regressive errors are decided by this program, not by the analysis
code, and written to the `error_type` column of `_behav.csv` (`ErrorClassifier` in
`prl_task.py`, lines 212–228).

- when a reversal passes, the criterion is reset
- the criterion is met the first time the new correct option is chosen
- an error **before** the criterion is met is `perseverative`, **after** it is
  `regressive`
- errors in the first segment (segment 0) are `acquisition`, and non-responses are
  `none`

Reapplying this rule to the whole raw dataset and comparing against the `error_type`
column gave no disagreement across 1,418 trials.

## Running it

```
"C:\Program Files\PsychoPy\python.exe" prl_task.py
```

To use the Cortex path, `cortex_credentials.json` must sit in the same folder.

```json
{"client_id": "...", "client_secret": "...", "license": ""}
```

That file is listed in `.gitignore`. Do not commit it.

## Randomisation

`seed = 0` is fixed, so every participant experienced the same trial order. The
reversals fall on trials 19, 36, 58, 75 and 98.

## Trial schedule

Publishing this is a commitment made in §38-2 of the preregistration.

| File | Contents |
|---|---|
| `make_schedule.py` | regenerates the seed-0 schedule and writes it to CSV. Runs without PsychoPy |
| `seed0_schedule.csv` | that output: segment, correct option, reward assignment and left/right position for all 120 trials |

```bash
python task/make_schedule.py
```

The generation rule is not restated here. `make_schedule.py` imports `build_trials`
from `prl_task.py` directly — if the rule were written in two places, the two would
eventually disagree.

**Because every participant received the same schedule generated from the same seed
(0), this single CSV fully specifies the stimulus order that every participant saw.**
If the raw data are present, `make_schedule.py` also checks the schedule against the
actual sessions (all 12 sessions confirmed to match).

```
segment lengths    19 · 17 · 22 · 17 · 23 · 22   (120 total)
reversal trials    19 · 36 · 58 · 75 · 98
first correct      A
rewards scheduled  89 of 120 trials = 74.2%
```

Note the last line. The setting is `p_reward = 0.80`, but **the rate realised in this
fixed random sequence is 74.2%**, and because the seed is fixed every participant
experienced that rate.
