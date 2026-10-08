"""
prl_task.py  (v2)
Probabilistic reversal-learning (PRL) task for autistic adults, with resting-state EEG and Cortex marker synchronisation.

Changes from v1
--------------
1. Practice split into two stages
   - stage 1: 100% contingency. Only the stimulus-outcome mapping has to be learned.
   - stage 2: 80/20 with one reversal. Probabilistic errors and a rule change are experienced in advance.
2. Instructions split over several short screens, plus a comprehension check
3. Feedback drawn as O / X shapes (no font dependency)
4. Larger stimuli and text, with an explicit font for Korean
5. Shorter inter-trial times (about 2 minutes off the total)
6. Progress and cumulative score shown between blocks

Running it
----
With PILOT set to True in CFG it runs short and without EEG. Set it back to False before the real session.
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import sys

# The PsychoPy Coder reads child-process output as UTF-8.
# That clashes with the CP949 default of a Korean Windows install and hangs the Coder, so it is forced here.
# (Console output is written in ASCII only. The on-screen instruction text is a separate matter.)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from psychopy import core, event, gui, logging, monitors, visual

# PsychoPy's font manager walks every system font at start-up and emits warnings.
# A flood of warnings down the console pipe slows the Coder noticeably, so they are suppressed.
# (Real errors still show.)
logging.console.setLevel(logging.ERROR)

# --- EEG backend ---
# "lsl"    : EmotivPRO (Standard or above) + Lab Streaming Layer  <- recommended
# "cortex" : Cortex API injectMarker (needs a Developer API licence)
BACKEND = "lsl"

if BACKEND == "lsl":
    from lsl_client import get_client
else:
    from cortex_client import get_client

# ==================================================================== #
# configuration
# ==================================================================== #
PILOT = False          # True = short, no EEG (dry run). Set back to False before the real session.

CFG = dict(
    # --- screen ---
    fullscr=True,
    screen=0,
    bg_color=[0.0, 0.0, 0.0],
    monitor_name="testMonitor",
    monitor_width_cm=34.5,
    monitor_dist_cm=60,
    font="Malgun Gothic",            # for displaying Korean. Change to "Gulim" if it renders wrong

    # --- resting state ---
    rest_condition="EC",
    rest_duration=90.0,
    rest_runs=2,
    rest_gap=20.0,                   # seconds between resting runs. Advances automatically, no key press

    # --- main task ---
    n_trials=120,
    n_blocks=3,
    n_reversals=5,
    p_reward=0.80,
    seg_min=17,
    seg_max=23,
    guard_trials=3,

    # --- practice ---
    prac1_max=24,                    # maximum trials in practice stage 1 (100% contingency)
    prac1_criterion=7,               # this many correct in a row passes
    prac1_attempts=2,                # retries after re-explaining when the criterion is not met
    prac2_trials=12,                 # practice stage 2: 80/20 with one reversal partway

    # --- timing (shorter than v1) ---
    fix_min=0.400,
    fix_max=0.700,
    resp_max=4.000,
    choice_hl=0.200,
    fb_dur=0.700,
    iti=0.500,
    block_gap=2.000,                 # blank screen between blocks. Automatic, no rest period

    # --- sizes (larger than v1) ---
    stim_size=7.0,                   # deg. v1 was 4.0
    stim_offset=9.5,                 # deg. v1 was 7.0
    fb_size=5.0,                     # O/X size
    text_h=1.4,                      # instruction text. v1 was 0.9
    fix_h=2.0,
    wrap=26.0,
)

if PILOT:
    CFG["rest_duration"] = 10.0      # shorter resting state for the dry run
    CFG["rest_gap"] = 3.0

# marker code table
# Only the integers below survive in the EmotivPRO recording. The labels are stored in the local _markers.csv only.
#   1  session_start      30  fix_on
#   2  session_end        31  stim_on
#  10  rest1_start        32  response
#  11  rest1_end          33  feedback X
#  12  rest2_start        34  feedback O
#  13  rest2_end          39  timeout
#  20  block1_start       40  reversal
#  21  block2_start       98  sync_begin
#  22  block3_start       99  sync_end
#  25  block_end
M = dict(
    session_start=("session_start", 1), session_end=("session_end", 2),
    rest_start=("rest_start", 10), rest_end=("rest_end", 11),
    block_start=("block_start", 20), block_end=("block_end", 25),
    fix_on=("fix_on", 30), stim_on=("stim_on", 31),
    response=("response", 32), fb_on=("fb_on", 33), timeout=("timeout", 39),
    reversal=("reversal", 40), sync=("sync", 99),
)

# ==================================================================== #
# clock alignment  (same as v1)
# ==================================================================== #
def measure_epoch_offset(n=25):
    offs = []
    for _ in range(n):
        m0 = core.monotonicClock.getTime()
        e = time.time()
        m1 = core.monotonicClock.getTime()
        offs.append(e - (m0 + m1) / 2.0)
    offs.sort()
    return offs[len(offs) // 2]


EPOCH_OFFSET = None


def to_epoch_ms(mono_t):
    return (mono_t + EPOCH_OFFSET) * 1000.0


# ==================================================================== #
# reversal schedule
# ==================================================================== #
def build_schedule(cfg, rng):
    n_seg = cfg["n_reversals"] + 1
    bl = cfg["n_trials"] // cfg["n_blocks"]
    boundaries = [bl * i for i in range(1, cfg["n_blocks"])]
    for _ in range(20000):
        segs = [rng.randint(cfg["seg_min"], cfg["seg_max"]) for _ in range(n_seg)]
        if sum(segs) != cfg["n_trials"]:
            continue
        rev_at, acc = [], 0
        for s in segs[:-1]:
            acc += s
            rev_at.append(acc)
        if any(abs(r - b) <= cfg["guard_trials"] for r in rev_at for b in boundaries):
            continue
        return segs, rev_at
    raise RuntimeError("failed to build the reversal schedule")


def build_trials(cfg, rng):
    segs, rev_at = build_schedule(cfg, rng)
    target, trials, t = "A", [], 0
    for si, seg_len in enumerate(segs):
        for k in range(seg_len):
            trials.append(dict(
                trial=t, block=t // (cfg["n_trials"] // cfg["n_blocks"]),
                segment=si, target=target, trial_in_seg=k,
                is_reversal_trial=int(k == 0 and si > 0),
                rewarded_if_correct=int(rng.random() < cfg["p_reward"]),
                left_stim=rng.choice(["A", "B"]),
            ))
            t += 1
        target = "B" if target == "A" else "A"
    return trials, rev_at


def build_practice2(cfg, rng, first_target):
    """Practice stage 2: 80/20 contingency with one reversal partway."""
    out = []
    half = cfg["prac2_trials"] // 2
    other = "B" if first_target == "A" else "A"
    for i in range(cfg["prac2_trials"]):
        tgt = first_target if i < half else other
        out.append(dict(phase=2, target=tgt,
                        rewarded_if_correct=int(rng.random() < cfg["p_reward"]),
                        left_stim=rng.choice(["A", "B"]),
                        is_reversal_trial=int(i == half)))
    return out


# ==================================================================== #
# error-type classification
# ==================================================================== #
class ErrorClassifier:
    def __init__(self):
        self.segment = None
        self.criterion_met = False

    def classify(self, tr, choice):
        if tr["segment"] != self.segment:
            self.segment = tr["segment"]
            self.criterion_met = False
        if choice is None:
            return "none"
        if choice == tr["target"]:
            self.criterion_met = True
            return "correct"
        if tr["segment"] == 0:
            return "acquisition"
        return "regressive" if self.criterion_met else "perseverative"


# ==================================================================== #
# display
# ==================================================================== #
class Display:
    def __init__(self, cfg):
        mon = monitors.Monitor(cfg["monitor_name"])
        mon.setWidth(cfg["monitor_width_cm"])
        mon.setDistance(cfg["monitor_dist_cm"])

        self.win = visual.Window(
            fullscr=cfg["fullscr"], screen=cfg["screen"], monitor=mon,
            color=cfg["bg_color"], colorSpace="rgb", units="deg",
            allowGUI=False, waitBlanking=True)
        self.win.mouseVisible = False
        self.cfg = cfg

        s, off = cfg["stim_size"], cfg["stim_offset"]
        self.stim = {
            "A": visual.Circle(self.win, radius=s / 2,
                               fillColor=[-0.4, 0.1, 0.7], lineColor=None),
            "B": visual.Rect(self.win, width=s, height=s,
                             fillColor=[0.85, 0.35, -0.6], lineColor=None),
        }
        self.frame = visual.Rect(self.win, width=s * 1.3, height=s * 1.3,
                                 lineColor=[0.95, 0.95, 0.95], lineWidth=8,
                                 fillColor=None)
        self.pos = {"left": (-off, 0), "right": (off, 0)}

        # --- O / X drawn as shapes, so no font dependency ---
        f = cfg["fb_size"]
        self.fb_o = visual.Circle(self.win, radius=f / 2, fillColor=None,
                                  lineColor=[-0.3, 0.75, 0.1], lineWidth=14)
        self.fb_x = [
            visual.Line(self.win, start=(-f / 2.6, -f / 2.6), end=(f / 2.6, f / 2.6),
                        lineColor=[0.85, -0.35, -0.35], lineWidth=14),
            visual.Line(self.win, start=(-f / 2.6, f / 2.6), end=(f / 2.6, -f / 2.6),
                        lineColor=[0.85, -0.35, -0.35], lineWidth=14),
        ]

        self.fix = visual.TextStim(self.win, text="+", height=cfg["fix_h"],
                                   color=[0.75, 0.75, 0.75], font=cfg["font"])
        self.msg = visual.TextStim(self.win, text="", height=cfg["text_h"],
                                   color=[0.9, 0.9, 0.9], wrapWidth=cfg["wrap"],
                                   font=cfg["font"], alignText="center")

    def draw_pair(self, left_stim, highlight=None):
        right_stim = "B" if left_stim == "A" else "A"
        for side, name in (("left", left_stim), ("right", right_stim)):
            self.stim[name].pos = self.pos[side]
            self.stim[name].draw()
        if highlight is not None:
            self.frame.pos = self.pos[highlight]
            self.frame.draw()
        return right_stim

    def draw_fb(self, correct):
        if correct:
            self.fb_o.draw()
        else:
            for ln in self.fb_x:
                ln.draw()

    def text_screen(self, txt, wait_key=True, keys=("space",)):
        self.msg.text = txt
        self.msg.draw()
        self.win.flip()
        if wait_key:
            k = event.waitKeys(keyList=list(keys) + ["escape"])
            if "escape" in k:
                raise KeyboardInterrupt
            return k[0]

    def hold(self, dur, draw_fn=None):
        t0 = self.win.flip() if draw_fn is None else None
        if draw_fn is not None:
            draw_fn()
            t0 = self.win.flip()
        while core.monotonicClock.getTime() - t0 < dur:
            if draw_fn is not None:
                draw_fn()
            self.win.flip()
        return t0

    def close(self):
        self.win.mouseVisible = True
        self.win.close()


# ==================================================================== #
# resting state
# ==================================================================== #
def run_rest(disp, cortex, cfg, run_idx):
    if cfg["rest_condition"] == "EC":
        # Verbatim Korean as shown to the participant. English: "Time for a short
        # break now. Please close your eyes and sit comfortably. There is nothing in
        # particular you need to think about. I will tell you when it is over.
        # Press the space bar when you are ready."
        instr = ("이제 잠시 쉬는 시간입니다.\n\n"
                 "눈을 감고 편하게 계셔 주세요.\n"
                 "특별히 무언가를 생각하실 필요는 없습니다.\n\n"
                 "끝나면 제가 말씀드리겠습니다.\n\n"
                 "준비되시면 스페이스바를 눌러 주세요.")
    else:
        # Verbatim Korean as shown to the participant. English: "Please look
        # comfortably at the cross in the middle of the screen. Press the space bar
        # when you are ready."
        instr = ("화면 가운데 십자를 편하게 바라봐 주세요.\n\n"
                 "준비되시면 스페이스바를 눌러 주세요.")
    disp.text_screen(instr)

    disp.fix.draw()
    t0 = disp.win.flip()
    # Keep the value distinct per run (only integers survive in the EmotivPRO recording)
    cortex.mark(f"rest{run_idx}_start",
                M["rest_start"][1] + (run_idx - 1) * 2, to_epoch_ms(t0))
    while core.monotonicClock.getTime() - t0 < cfg["rest_duration"]:
        if event.getKeys(keyList=["escape"]):
            raise KeyboardInterrupt
        disp.fix.draw()
        disp.win.flip()
    disp.fix.draw()
    t1 = disp.win.flip()
    cortex.mark(f"rest{run_idx}_end",
                M["rest_end"][1] + (run_idx - 1) * 2, to_epoch_ms(t1))
    return dict(run=run_idx, condition=cfg["rest_condition"],
                onset_epoch_ms=to_epoch_ms(t0), offset_epoch_ms=to_epoch_ms(t1),
                duration_s=t1 - t0)


# ==================================================================== #
# trial
# ==================================================================== #
def run_trial(disp, cortex, cfg, tr, mouse, rng, practice=False):
    # fixation
    fix_dur = rng.uniform(cfg["fix_min"], cfg["fix_max"])
    disp.fix.draw()
    t_fix = disp.win.flip()
    if not practice:
        cortex.mark("fix_on", M["fix_on"][1], to_epoch_ms(t_fix))
    while core.monotonicClock.getTime() - t_fix < fix_dur:
        disp.fix.draw()
        disp.win.flip()

    # stimuli
    right_stim = disp.draw_pair(tr["left_stim"])
    mouse.clickReset()
    event.clearEvents()
    t_stim = disp.win.flip()
    if not practice:
        cortex.mark("stim_on", M["stim_on"][1], to_epoch_ms(t_stim))

    choice_side, rt = None, None
    while core.monotonicClock.getTime() - t_stim < cfg["resp_max"]:
        if event.getKeys(keyList=["escape"]):
            raise KeyboardInterrupt
        buttons, times = mouse.getPressed(getTime=True)
        if buttons[0]:
            choice_side, rt = "left", times[0]
            break
        if buttons[2]:
            choice_side, rt = "right", times[2]
            break
        disp.draw_pair(tr["left_stim"])
        disp.win.flip()

    if choice_side is None:
        t_to = disp.win.flip()
        if not practice:
            cortex.mark("timeout", M["timeout"][1], to_epoch_ms(t_to))
        # Verbatim Korean as shown to the participant. English: "Please choose a little faster."
        disp.msg.text = "조금 더 빠르게 선택해 주세요."
        disp.hold(1.2, draw_fn=disp.msg.draw)
        return dict(choice=None, choice_side=None, rt=None, correct=None,
                    reward=None, stim_onset_epoch_ms=to_epoch_ms(t_stim),
                    resp_epoch_ms=None, timeout=1)

    resp_epoch_ms = to_epoch_ms(t_stim + rt)
    if not practice:
        cortex.mark("response", M["response"][1], resp_epoch_ms)

    chosen = tr["left_stim"] if choice_side == "left" else right_stim
    correct = int(chosen == tr["target"])
    reward = int(correct and tr["rewarded_if_correct"])

    # highlight the choice
    disp.hold(cfg["choice_hl"],
              draw_fn=lambda: disp.draw_pair(tr["left_stim"], highlight=choice_side))

    # feedback (O / X)
    disp.draw_fb(reward)
    t_fb = disp.win.flip()
    if not practice:
        cortex.mark("fb_on", M["fb_on"][1] + reward, to_epoch_ms(t_fb))
    while core.monotonicClock.getTime() - t_fb < cfg["fb_dur"]:
        disp.draw_fb(reward)
        disp.win.flip()

    # ITI
    disp.hold(cfg["iti"])

    return dict(choice=chosen, choice_side=choice_side, rt=rt, correct=correct,
                reward=reward, stim_onset_epoch_ms=to_epoch_ms(t_stim),
                resp_epoch_ms=resp_epoch_ms, timeout=0)


# ==================================================================== #
# instructions and practice
# ==================================================================== #
# Verbatim Korean as shown to the participant. English, screen by screen:
#   1. "We are going to play a simple game. A blue circle and an orange square
#      will appear side by side on the screen. Press the space bar."
#   2. "Choose one of the two with the mouse. Left button = left picture,
#      right button = right picture. Press the space bar."
#   3. "When you choose, an O or an X will appear. The goal is to get as many Os as
#      you can. First we will practise. Press the space bar."
INSTR = [
    "지금부터 간단한 게임을 하나 하겠습니다.\n\n"
    "화면에 파란 동그라미와 주황 네모가\n나란히 나타납니다.\n\n"
    "스페이스바를 눌러 주세요.",

    "둘 중 하나를 마우스로 고르시면 됩니다.\n\n"
    "왼쪽 버튼 = 왼쪽 그림\n"
    "오른쪽 버튼 = 오른쪽 그림\n\n"
    "스페이스바를 눌러 주세요.",

    "고르시면 O 또는 X 가 나타납니다.\n\n"
    "O 를 많이 받는 것이 목표입니다.\n\n"
    "먼저 연습을 해 보겠습니다.\n\n"
    "스페이스바를 눌러 주세요.",
]

# Shown again when the criterion is not met. English: "Let me say it once more.
# One of the two pictures gives an O and the other gives an X. Find the picture
# that gives an O and keep choosing it. Press the space bar."
REEXPLAIN = (
    "다시 한 번 말씀드리겠습니다.\n\n"
    "두 그림 중 하나는 O 를 주고,\n다른 하나는 X 를 줍니다.\n\n"
    "O 를 주는 그림을 찾아 계속 고르시면 됩니다.\n\n"
    "스페이스바를 눌러 주세요."
)

# Verbatim Korean as shown to the participant. English, screen by screen:
#   1. "Well done. This time it is a little different. Even when you choose the
#      good picture, an X will sometimes come up. Press the space bar."
#   2. "And partway through, the good picture may change. Keep going and choose
#      whichever side gives an O more often. Press the space bar."
INSTR2 = [
    "잘하셨습니다.\n\n"
    "이번에는 조금 다릅니다.\n\n"
    "좋은 그림을 골라도\n가끔 X 가 나올 수 있습니다.\n\n"
    "스페이스바를 눌러 주세요.",

    "그리고 중간에\n좋은 그림이 바뀔 수도 있습니다.\n\n"
    "계속 해 보시면서\nO 가 더 자주 나오는 쪽을 고르시면 됩니다.\n\n"
    "스페이스바를 눌러 주세요.",
]


class PracticeFailed(Exception):
    """Raised when practice stage 1 does not reach the criterion."""


def run_practice1(disp, cortex, cfg, mouse, rng, target, attempt):
    """Practice stage 1: 100% contingency. Passes as soon as the run-of-correct criterion is reached.

    Returns (passed, trials used, trials to criterion or None)
    """
    run_streak, n_done, reached_at = 0, 0, None
    for i in range(cfg["prac1_max"]):
        tr = dict(phase=1, target=target, rewarded_if_correct=1,
                  left_stim=rng.choice(["A", "B"]), is_reversal_trial=0)
        res = run_trial(disp, cortex, cfg, tr, mouse, rng, practice=True)
        n_done += 1
        if res["correct"] == 1:
            run_streak += 1
        else:
            run_streak = 0
        if run_streak >= cfg["prac1_criterion"]:
            reached_at = n_done
            return True, n_done, reached_at
    return False, n_done, None


def run_practice(disp, cortex, cfg, mouse, rng):
    """The whole practice. Rather than a verbal comprehension check, the ability to perform is judged behaviourally.

    If learning does not happen under a 100% contingency it will be harder still under 80/20,
    so reaching the stage-1 criterion is a direct check that the task can be performed.
    """
    target = rng.choice(["A", "B"])
    for s in INSTR:
        disp.text_screen(s)

    log = []
    passed = False
    for attempt in range(1, cfg["prac1_attempts"] + 1):
        if attempt > 1:
            disp.text_screen(REEXPLAIN)
        ok, n_done, reached = run_practice1(
            disp, cortex, cfg, mouse, rng, target, attempt)
        log.append(dict(attempt=attempt, n_trials=n_done,
                        criterion_met=int(ok), trials_to_criterion=reached))
        if ok:
            passed = True
            break

    if not passed:
        # Verbatim Korean as shown to the participant. English: "Thank you for your
        # effort. We will stop here. Thank you. Press the space bar."
        disp.text_screen("수고하셨습니다.\n\n"
                         "여기까지 하겠습니다.\n"
                         "감사합니다.\n\n"
                         "스페이스바를 눌러 주세요.")
        raise PracticeFailed(json.dumps(log, ensure_ascii=False))

    # --- stage 2: probabilistic feedback plus a reversal ---
    for s in INSTR2:
        disp.text_screen(s)
    for tr in build_practice2(cfg, rng, target):
        run_trial(disp, cortex, cfg, tr, mouse, rng, practice=True)

    # Verbatim Korean as shown to the participant. English: "Practice is over. If you
    # have any questions, please ask now. We will start the real game. It runs
    # through to the end in one go. Press the space bar when you are ready."
    disp.text_screen("연습이 끝났습니다.\n\n"
                     "궁금한 점이 있으시면\n지금 말씀해 주세요.\n\n"
                     "이제 본 게임을 시작하겠습니다.\n"
                     "한 번에 끝까지 이어서 진행됩니다.\n\n"
                     "준비되시면 스페이스바를 눌러 주세요.")
    return dict(practice_target=target, phase1=log, phase1_passed=1)


# ==================================================================== #
# main
# ==================================================================== #
def main():
    global EPOCH_OFFSET

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eeg", action="store_true", default=PILOT)
    ap.add_argument("--skip-rest", action="store_true")
    ap.add_argument("--windowed", action="store_true")
    args, _ = ap.parse_known_args()          # ignore the options the PsychoPy Runner appends

    cfg = dict(CFG)
    if args.windowed:
        cfg["fullscr"] = False

    # --- bring up the clock offset and the EEG client before the dialog ---
    # The LSL outlet has to be up first for the stream to appear in EmotivPRO's Inlet list.
    EPOCH_OFFSET = measure_epoch_offset()
    offset_start = EPOCH_OFFSET

    cortex = get_client(use_eeg=not args.no_eeg)
    cortex.connect()

    if BACKEND == "lsl" and not args.no_eeg:
        print("\n" + "=" * 58)
        print(" In EmotivPRO, complete these steps, then press OK:")
        print("   1. Start Record")
        print("   2. Settings > Lab Streaming Layer > Inlet")
        print("   3. Select 'PRL_Markers' and click Connect")
        print("=" * 58 + "\n")

    info = {"participant ID": "ASD", "session note": "", "seed": 0}
    order = ["participant ID", "session note", "seed"]
    if BACKEND == "lsl" and not args.no_eeg:
        info["recording started + Inlet connected"] = False
        order.insert(0, "recording started + Inlet connected")
    dlg = gui.DlgFromDict(info, title="PRL task", order=order)
    if not dlg.OK:
        cortex.close()
        core.quit()
    if BACKEND == "lsl" and not args.no_eeg \
            and not info.get("recording started + Inlet connected"):
        print("\n[ABORT] Start recording and connect the Inlet first.")
        cortex.close()
        core.quit()

    pid = info["participant ID"].strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("data", pid)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{pid}_{stamp}")

    seed = int(info["seed"]) if str(info["seed"]).strip() else int(time.time()) % 100000
    rng = random.Random(seed)

    rec_id = cortex.start_record(title=f"{pid}_{stamp}",
                                description="PRL + resting state")

    disp = Display(cfg)
    mouse = event.Mouse(win=disp.win, visible=False)
    trials, rev_at = build_trials(cfg, rng)
    clf = ErrorClassifier()

    rows, rest_log = [], []
    practice_log = None
    n_o = 0
    try:
        t_ss = disp.win.flip()
        cortex.mark("session_start", M["session_start"][1], to_epoch_ms(t_ss))
        cortex.mark("sync_begin", M["sync"][1] - 1, to_epoch_ms(t_ss))

        if not args.skip_rest:
            for r in range(1, cfg["rest_runs"] + 1):
                rest_log.append(run_rest(disp, cortex, cfg, r))
                if r < cfg["rest_runs"]:
                    # The participant's eyes are closed, so no key press is asked for.
                    # The experimenter says so aloud and the screen keeps the fixation cross.
                    disp.hold(cfg["rest_gap"], draw_fn=disp.fix.draw)

        practice_log = run_practice(disp, cortex, cfg, mouse, rng)

        per_block = cfg["n_trials"] // cfg["n_blocks"]
        cur_block = -1
        for tr in trials:
            if tr["block"] != cur_block:
                if cur_block >= 0:
                    t_be = disp.win.flip()
                    cortex.mark("block_end", M["block_end"][1], to_epoch_ms(t_be))
                    # Automatic transition with no rest period, to avoid posture changes and artefacts.
                    # The screen is briefly blanked to break the flow of trials.
                    disp.hold(cfg["block_gap"], draw_fn=disp.fix.draw)
                cur_block = tr["block"]
                t_bs = disp.win.flip()
                cortex.mark("block_start", M["block_start"][1] + cur_block,
                            to_epoch_ms(t_bs))

            if tr["is_reversal_trial"]:
                cortex.mark("reversal", M["reversal"][1],
                            to_epoch_ms(core.monotonicClock.getTime()))

            res = run_trial(disp, cortex, cfg, tr, mouse, rng)
            n_o += (res["reward"] or 0)
            rows.append({**tr, **res, "error_type": clf.classify(tr, res["choice"])})

        t_be = disp.win.flip()
        cortex.mark("block_end", M["block_end"][1], to_epoch_ms(t_be))
        t_se = disp.win.flip()
        cortex.mark("sync_end", M["sync"][1], to_epoch_ms(t_se))
        cortex.mark("session_end", M["session_end"][1], to_epoch_ms(t_se))

        # Verbatim Korean as shown to the participant. English: "That is everything.
        # Thank you for your effort. Press the space bar."
        disp.text_screen("모두 끝났습니다.\n\n"
                         "수고하셨습니다.\n\n"
                         "스페이스바를 눌러 주세요.")

    except PracticeFailed as e:
        practice_log = dict(phase1=json.loads(str(e)), phase1_passed=0)
        print("\n[STOP] Practice phase 1 criterion not met - main task skipped")

    except KeyboardInterrupt:
        print("\n[STOP] Terminated by user")
    finally:
        if rows:
            with open(base + "_behav.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        offset_end = measure_epoch_offset()
        meta = dict(participant=pid, timestamp=stamp, seed=seed,
                    note=info["session note"], record_id=rec_id, pilot=PILOT,
                    backend=(BACKEND if not args.no_eeg else "none"),
                    epoch_offset_start=offset_start, epoch_offset_end=offset_end,
                    epoch_offset_drift_ms=(offset_end - offset_start) * 1000.0,
                    reversal_trials=rev_at, n_trials_completed=len(rows),
                    total_rewards=n_o, rest=rest_log, practice=practice_log,
                    config=cfg)
        with open(base + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        with open(base + "_markers.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["label", "value", "epoch_ms"])
            w.writerows(cortex.marker_log)

        try:
            disp.close()
        except Exception:
            pass
        cortex.stop_record()
        cortex.close()
        print(f"\nSaved: {base}_*.csv / _meta.json")
        print(f"Clock drift: {meta['epoch_offset_drift_ms']:.2f} ms")
        core.quit()


if __name__ == "__main__":
    main()
