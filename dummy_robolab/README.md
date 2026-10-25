# Local RoboLab test stand-in

This folder is test code. Production `bridge.py` and `cameras.py` do not import it.

`test_nuc_executor.py` exercises the standalone production `nuc_executor.py`
with a dummy DROID robot over loopback TCP, including two successive chunks
from the actual bridge. It tests 15 Hz wall-clock dispatch, measured feedback,
gripper mapping, no commands during inference, busy request rejection, failure
handling and disconnects. An optional ZeroRPC test runs the actual executor CLI
with the unchanged DROID client and a local dummy service on port 4242. Neither
test initializes or connects to a physical robot. Run the tests without Isaac:

```bash
.venv/bin/python -m unittest discover -s dummy_robolab -p 'test_*.py' -v
```

`sim_nuc.py` uses real RoboLab/Isaac cameras and physics. It listens only on a
random `127.0.0.1` TCP port and implements the minimal bridge's NUC protocol.
`client.py` runs the actual bridge, replacing only `ZedCameras` with on-request
Isaac RGB captures. State still arrives through TCP. Cosmos remains the real
official server at `ws://127.0.0.1:8000/`. All three views are compared with the
reference RoboLab Cosmos client before inference. No physical NUC is contacted.

With RoboLab installed at `~/Documents/RoboLab`, from this project's root:

```bash
# One-time Isaac Lab 2.2 / R580 compatibility dependency in an isolated test cache.
uv pip install --python ~/Documents/RoboLab/.venv/bin/python \
  --target ~/.cache/cosmos-robolab-warp-1.8.1 --no-deps 'warp-lang==1.8.1'

# The official Cosmos policy server must already be running on localhost:8000.
.venv/bin/python dummy_robolab/run.py
```

The runner first executes `test_protocol.py`: stale/flooded observations, busy
replies, fragmented TCP, early task completion, partial terminal chunks, bad
completion IDs, queued/false completion, missing ACKs, disconnects, malformed
JSON, invalid state, camera failure, and invalid policy responses. These failure
tests use camera/model stubs and a loopback TCP peer.

The full task then uses actual Cosmos inference and simulated physics. Extra
state packets are sent during inference and every executed action to check that
they get `busy` replies and never trigger extra inference. Old request IDs are
also injected deliberately. The verifier checks exact full-output delivery,
action order, measured feedback, fresh camera captures, and immediate termination.

Outputs are in `outputs/robolab_*`; `outputs/robolab_latest.txt` points to the latest
run. Each includes `result.json`, protocol logs, model inputs/outputs, TCP chunks,
per-step measured states, and `simulator/rollout.mp4`.

Simulation uses 15 Hz in simulation time and pauses physics during inference.
It does not validate real USB access, physical controller behavior, or network
timing. The normal bridge uses the real local UVC cameras when launched directly.
