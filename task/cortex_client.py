"""
cortex_client.py
A minimal client for the Emotiv Cortex API (v2).

Design principle
---------
The accuracy of a marker is set not by when it was sent but by the *event* time carried in
the `time` field of injectMarker. The stimulus-presentation thread therefore only stamps the
time, puts it on a queue and returns immediately; a background worker does the actual
WebSocket send. So network and JSON-RPC delays never affect the marker timestamp.

Before running
---------
1. Start EMOTIV Launcher and log in (the Cortex service listens on wss://localhost:6868)
2. Register an app at https://www.emotiv.com/developer to get a client_id and client_secret
3. Write a cortex_credentials.json file:
   {"client_id": "...", "client_secret": "...", "license": ""}
4. pip install websocket-client
"""

import json
import os
import queue
import ssl
import threading
import time

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover
    websocket = None

CORTEX_URL = "wss://localhost:6868"
CRED_FILE = "cortex_credentials.json"


class CortexError(RuntimeError):
    pass


class CortexClient:
    """Cortex session management plus non-blocking marker injection."""

    def __init__(self, cred_path=CRED_FILE, verbose=True):
        self.verbose = verbose
        self._req_id = 0
        self._lock = threading.Lock()
        self.ws = None
        self.token = None
        self.session_id = None
        self.record_id = None
        self.headset_id = None

        with open(cred_path, "r", encoding="utf-8") as f:
            cred = json.load(f)
        self.client_id = cred["client_id"]
        self.client_secret = cred["client_secret"]
        self.license = cred.get("license", "")

        # marker queue and worker
        self._mq = queue.Queue()
        self._stop = threading.Event()
        self._worker = None
        self.marker_log = []  # local copy (for checking against the Cortex recording)

    # ------------------------------------------------------------------ #
    # low-level RPC
    # ------------------------------------------------------------------ #
    def _log(self, *a):
        if self.verbose:
            print("[cortex]", *a)

    def _rpc(self, method, params, timeout=10):
        with self._lock:
            self._req_id += 1
            rid = self._req_id
            msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
            self.ws.send(json.dumps(msg))
            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = self.ws.recv()
                resp = json.loads(raw)
                if resp.get("id") == rid:
                    if "error" in resp:
                        raise CortexError(f"{method}: {resp['error']}")
                    return resp["result"]
                # warning and stream messages are ignored
            raise CortexError(f"{method}: timeout")

    # ------------------------------------------------------------------ #
    # connection sequence
    # ------------------------------------------------------------------ #
    def connect(self):
        if websocket is None:
            raise CortexError("websocket-client is not installed: pip install websocket-client")

        self.ws = websocket.create_connection(
            CORTEX_URL, sslopt={"cert_reqs": ssl.CERT_NONE}, timeout=15
        )
        self._log("connected")

        # 1) access rights
        r = self._rpc("requestAccess",
                      {"clientId": self.client_id, "clientSecret": self.client_secret})
        if not r.get("accessGranted"):
            raise CortexError(
                "App access not granted. Approve this app in EMOTIV Launcher and run again."
            )

        # 2) token
        params = {"clientId": self.client_id, "clientSecret": self.client_secret,
                  "debit": 1}
        if self.license:
            params["license"] = self.license
        self.token = self._rpc("authorize", params)["cortexToken"]
        self._log("authorised")

        # 3) headset
        headsets = self._rpc("queryHeadsets", {})
        if not headsets:
            raise CortexError("No headset found. Check the dongle and the power.")
        hs = headsets[0]
        self.headset_id = hs["id"]
        self._log(f"headset: {self.headset_id} (status={hs.get('status')})")

        if hs.get("status") != "connected":
            self._rpc("controlDevice",
                      {"command": "connect", "headset": self.headset_id}, timeout=20)
            time.sleep(2)

        # 4) session
        self.session_id = self._rpc("createSession", {
            "cortexToken": self.token,
            "headset": self.headset_id,
            "status": "active",
        })["id"]
        self._log(f"session: {self.session_id}")

        # start the worker
        self._worker = threading.Thread(target=self._marker_worker, daemon=True)
        self._worker.start()
        return self

    # ------------------------------------------------------------------ #
    # recording
    # ------------------------------------------------------------------ #
    def start_record(self, title, description=""):
        r = self._rpc("createRecord", {
            "cortexToken": self.token,
            "session": self.session_id,
            "title": title,
            "description": description,
        })
        self.record_id = r["record"]["uuid"]
        self._log(f"recording started: {title}")
        return self.record_id

    def stop_record(self):
        if self.record_id is None:
            return None
        # Drain the queue first (injection is refused once the recording has ended)
        self.flush(timeout=5)
        r = self._rpc("stopRecord",
                      {"cortexToken": self.token, "session": self.session_id})
        self._log("recording ended")
        rid, self.record_id = self.record_id, None
        return rid

    def export_record(self, record_ids, folder, fmt="EDF",
                      stream_types=("EEG", "MOTION")):
        os.makedirs(folder, exist_ok=True)
        params = {
            "cortexToken": self.token,
            "recordIds": list(record_ids),
            "folder": os.path.abspath(folder),
            "format": fmt,
            "streamTypes": list(stream_types),
        }
        if fmt == "CSV":
            params["version"] = "V2"
        return self._rpc("exportRecord", params, timeout=120)

    # ------------------------------------------------------------------ #
    # markers
    # ------------------------------------------------------------------ #
    def mark(self, label, value, epoch_ms):
        """Non-blocking. Returns at once; the worker does the sending.

        epoch_ms : float  when the event *actually happened* (Unix epoch, ms)
        """
        self.marker_log.append((label, value, epoch_ms))
        self._mq.put((label, value, epoch_ms))

    def _marker_worker(self):
        while not self._stop.is_set():
            try:
                item = self._mq.get(timeout=0.2)
            except queue.Empty:
                continue
            label, value, epoch_ms = item
            try:
                self._rpc("injectMarker", {
                    "cortexToken": self.token,
                    "session": self.session_id,
                    "label": label,
                    "value": value,
                    "port": "psychopy",
                    "time": epoch_ms,     # <-- the field that sets the accuracy
                })
            except Exception as e:  # one failed marker does not stop the experiment
                self._log(f"marker failed {label}: {e}")
            finally:
                self._mq.task_done()

    def flush(self, timeout=5):
        t0 = time.time()
        while not self._mq.empty() and time.time() - t0 < timeout:
            time.sleep(0.05)

    def close(self):
        self.flush()
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2)
        try:
            if self.session_id:
                self._rpc("updateSession",
                          {"cortexToken": self.token,
                           "session": self.session_id, "status": "close"})
        except Exception:
            pass
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        self._log("closed")


class NullCortex:
    """A dummy for dry-running the task without EEG. Same interface."""

    def __init__(self, *a, **k):
        self.marker_log = []
        self.record_id = None

    def connect(self):
        print("[cortex] NULL mode - no EEG")
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
    return CortexClient(verbose=verbose) if use_eeg else NullCortex()
