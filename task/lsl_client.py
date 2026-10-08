"""
lsl_client.py
Marker transmission through EmotivPRO (Standard or above) + Lab Streaming Layer.

The interface matches cortex_client.py, so prl_task.py works unchanged.

Design principle
---------
The same as cortex_client. The accuracy of a marker is set by the *event* time, not the time
it was sent. LSL's push_sample takes an explicit timestamp, so the screen-flip time returned
by win.flip() is carried straight through.

LSL continuously estimates and corrects the clock offset between streams, which makes the
clock alignment one step safer than the Cortex route.

Before running
---------
1. Start EmotivPRO and log in (a Standard subscription or above is required)
2. Turn on the LSL feature in the EmotivPRO settings
3. Start [Record] in EmotivPRO first
   -- EmotivPRO accepts markers only while recording
4. In the LSL Inlet settings, select the MARKER_STREAM_NAME stream below and Connect
5. Run the task

For the marker value table see the M dictionary of prl_task.py.
Only the integer codes survive in the EmotivPRO recording; the label strings go to the local _markers.csv.

pylsl ships with PsychoPy Standalone by default.
If it is missing, install pylsl from the Plugin/packages manager.
"""

import threading
import time

try:
    import pylsl
except ImportError:  # pragma: no cover
    pylsl = None

MARKER_STREAM_NAME = "PRL_Markers"
MARKER_STREAM_TYPE = "Markers"
SOURCE_ID = "prl_task_marker_outlet"


class LSLError(RuntimeError):
    pass


class LSLClient:
    """The LSL marker outlet. EmotivPRO subscribes to this stream and inserts into the recording."""

    def __init__(self, verbose=True, stream_name=MARKER_STREAM_NAME):
        self.verbose = verbose
        self.stream_name = stream_name
        self.info = None
        self.outlet = None
        self.record_id = None
        self.marker_log = []          # local copy (for checking against the EmotivPRO recording)
        self._lock = threading.Lock()
        self._lsl_offset = None       # LSL clock - Unix epoch clock

    def _log(self, *a):
        if self.verbose:
            print("[lsl]", *a)

    # ------------------------------------------------------------------ #
    def _measure_offset(self, n=25):
        """Median of repeated measurements of the offset between pylsl.local_clock() and time.time()."""
        offs = []
        for _ in range(n):
            l0 = pylsl.local_clock()
            e = time.time()
            l1 = pylsl.local_clock()
            offs.append((l0 + l1) / 2.0 - e)
        offs.sort()
        return offs[len(offs) // 2]

    def connect(self):
        if pylsl is None:
            raise LSLError(
                "pylsl not installed. Install it via PsychoPy Plugin/packages manager.")

        # EmotivPRO requirement: channel_count must be 1 or 3.
        # The 3-channel form (MarkerTime / MarkerValue / CurrentTime) is used.
        # With the 1-channel form EmotivPRO takes the *receipt* time as the marker time, so the
        # transmission delay becomes error outright. The 3-channel form passes the event time
        # directly and lets EmotivPRO apply its synchronisation correction.
        info = pylsl.StreamInfo(
            name=self.stream_name,
            type=MARKER_STREAM_TYPE,
            channel_count=3,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_double64,
            source_id=SOURCE_ID,
        )
        chns = info.desc().append_child("channels")
        for label in ("MarkerTime", "MarkerValue", "CurrentTime"):
            chns.append_child("channel").append_child_value("label", label)
        info.desc().append_child_value("manufacturer", "PsychoPy")

        # Keep a reference to the StreamInfo.
        # Left as a local only, it is collected when the function returns, and liblsl can then
        # reference freed metadata when a subscriber connects, killing the process.
        self.info = info
        self.outlet = pylsl.StreamOutlet(info)
        self._lsl_offset = self._measure_offset()

        self._log(f"Marker outlet created: '{self.stream_name}' (3 channels)")
        self._log(f"LSL-epoch offset: {self._lsl_offset:.6f} s")
        self._log("In EmotivPRO: start Record, then Settings > LSL > Inlet > Connect.")

        # Give subscribers a moment to attach
        t0 = time.time()
        while time.time() - t0 < 1.5:
            if self.outlet.have_consumers():
                self._log("Consumer detected (EmotivPRO likely connected)")
                break
            time.sleep(0.1)
        else:
            self._log("!! No consumer detected yet. Check EmotivPRO LSL Inlet.")
            self._log("   (markers keep streaming but may not be recorded)")

        return self

    # ------------------------------------------------------------------ #
    # Recording is controlled from the EmotivPRO GUI. A no-op, to keep the interface.
    # ------------------------------------------------------------------ #
    def start_record(self, title, description=""):
        self.record_id = title
        self._log(f"Record id: {title} (recording controlled in EmotivPRO)")
        return title

    def stop_record(self):
        self._log("Task finished. Stop recording in EmotivPRO and export EDF.")
        rid, self.record_id = self.record_id, None
        return rid

    # ------------------------------------------------------------------ #
    def mark(self, label, value, epoch_ms):
        """Non-blocking. epoch_ms is when the event *actually happened* (Unix epoch, ms).

        Sent in the EmotivPRO 3-channel form.
          MarkerTime  : when the event occurred (epoch seconds)   <- this sets the accuracy
          MarkerValue : the integer code
          CurrentTime : when it is being pushed (epoch seconds)   <- for the synchronisation correction

        The label string is not sent to EmotivPRO and stays in the local CSV only.
        (EmotivPRO accepts integer markers only.)
        """
        self.marker_log.append((label, value, epoch_ms))
        marker_time = epoch_ms / 1000.0
        try:
            with self._lock:
                now = time.time()
                self.outlet.push_sample(
                    [marker_time, float(int(value)), now],
                    timestamp=marker_time + self._lsl_offset,
                    pushthrough=True)
        except Exception as e:
            self._log(f"marker failed {label}: {e}")

    def flush(self, timeout=5):
        pass          # push_sample sends immediately

    def close(self):
        self.outlet = None
        self.info = None
        self._log("closed")


class NullLSL:
    """A dummy for dry-running the task without EEG."""

    def __init__(self, *a, **k):
        self.marker_log = []
        self.record_id = None

    def connect(self):
        print("[lsl] NULL mode - no EEG")
        return self

    def start_record(self, title, description=""):
        return None

    def stop_record(self):
        return None

    def mark(self, label, value, epoch_ms):
        self.marker_log.append((label, value, epoch_ms))

    def flush(self, timeout=5):
        pass

    def close(self):
        pass


def get_client(use_eeg=True, verbose=True):
    return LSLClient(verbose=verbose) if use_eeg else NullLSL()
