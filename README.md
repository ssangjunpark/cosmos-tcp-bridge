# Minimal Cosmos TCP bridge

One bridge script plus a small `cameras.py` capture helper. The model stays in
the existing official Cosmos Framework server. No dependency on the larger
bridge project, RoboLab, or Isaac. Local RGB capture uses Linux UVC/V4L2 and
OpenCV factory rectification. The SDK is used only to enumerate camera serials.
`bridge.py` runs on the **5090 machine only**. The standalone `nuc_executor.py`
runs on the **NUC**, using its existing DROID controller APIs without changing
`robot.py`, `server_interface.py`, or the controller.

```text
NUC -> measured joints + gripper -> bridge <- local ZED cameras
                                  bridge -> Cosmos at ws://127.0.0.1:8000/
NUC <- entire policy output, including [32,8] actions <- bridge <- Cosmos
NUC -> execution completed -> bridge requests the next observation
```

The whole chunk travels in **one TCP message**. This script does not stream
individual actions or sleep at 15 Hz. The **NUC controller** executes the rows
at 15 Hz (32 actions span about 2.13 seconds).

## Install and run

```bash
cd ~/Documents/cosmos-tcp-bridge
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

Start the official Cosmos server separately, as before. Once the NUC executor
described below is listening, run on the 5090:

```bash
.venv/bin/python bridge.py --nuc-host NUC_IP --prompt 'Pick up the banana and place it in the bowl'
```

Default: keep running until stopped. Use `--chunks 1` for one chunk or
`--chunks 10` for ten chunks.
NUC port defaults to 9001 and Cosmos to `ws://127.0.0.1:8000/`.
Creating this project does not connect to the real NUC.

## Run the NUC executor

Copy just `nuc_executor.py` to the NUC. Run it with the NUC's existing DROID /
Polymetis Python environment, which already provides NumPy, ZeroRPC and gevent.
The 5090 requirements (Cosmos, ZED, etc.) are not needed on the NUC.

The local DROID API on port 4242 must already be initialized through the existing
robot setup. Starting `scripts/server/run_server.py` alone exposes the RPC API;
it does not initialize its robot/gripper interfaces. The executor connects with
`ServerInterface(ip_address="127.0.0.1", launch=False)` and checks readiness with
a state read before opening port 9001. It does not call launch, reset, or home.

On the NUC, after that existing setup is ready:

```bash
python /path/to/nuc_executor.py --droid-dir /path/to/droid --port 9001
```

`--droid-dir` is the checkout root containing `droid/misc/server_interface.py`;
omit it if DROID is already importable. `--host` defaults to `0.0.0.0` and
`--timeout` defaults to 60 seconds of TCP inactivity. The executor serves one
persistent bridge connection and exits when that connection closes. Restart it
explicitly for another session; it never reconnects or replays motion commands.

It handles the same two messages documented below:

- `get_observation`: call `get_robot_state()` now, map measured `joint_positions`
  and `gripper_position` to the policy keys, and include DROID's `timestamp`.
- `action_chunk`: validate the whole finite `[32,8]` chunk and 15 Hz rate, then
  call existing `update_command` with `action_space="joint_position"`,
  `gripper_action_space="position"`, and `blocking=False` for each row. Joint
  targets pass unchanged; the checkpoint gripper score becomes `float(score > 0.5)`.

Rows are dispatched about 66.7 ms apart, including the final row's interval.
If a command call exceeds that period or reports an error, remaining dispatches
stop and an error reply includes the returned-call count. Overlapping requests
receive `busy` and are discarded, including partial requests begun while busy.
Duplicate chunk IDs within a session are rejected. There is no pending-chunk queue.

The completion reply includes `"completion_basis":"dispatched"`. For wire
compatibility its `executed_actions` field counts returned `update_command`
calls. This is **not confirmation that every target was applied or reached**:
the unchanged DROID API uses background joint updates and can hide their errors.
The next observation is always new measured feedback. Tests verify the dispatch
contract; stronger physical completion reporting would require controller API work.

During inference, after the chunk limit, and after stopping the executor, no
additional targets are issued by this script. The running impedance controller
retains its last target and the gripper retains its target; the arm may continue
settling toward that target. Ctrl+C during a chunk sends a terminal `stopped`
reply with the dispatched prefix when the connection is available.

## NUC protocol: persistent TCP, newline-delimited JSON

The NUC listens for the bridge connection. Each request and response is one
JSON object followed by a newline. The existing robot controller remains on the
NUC; this project does not replace it. The NUC does not need to send images.

1. Bridge sends `{"type":"get_observation","request_id":"unique-session:observation:0"}`.
   The NUC samples its measured robot state and replies:

```python
{
    "status": "ok",
    "request_id": "unique-session:observation:0",
    "observation": {
        "observation/joint_position": q.tolist(),
        "observation/gripper_position": [closure],
    },
}
```

`q`: measured J1–J7 in radians.
`closure`: 0=open, 1=closed; the NUC converts native Robotiq feedback to this
convention. The prompt comes from the bridge command line. Immediately after
this state arrives, the bridge captures fresh local ZED images and packs RGB
uint8 `[540,640,3]`: wrist `[360,640]` on top, external-left and external-right
`[180,320]` below. UVC reads run concurrently, drain driver frames older than the
capture request, and keep no observation queue. Accepted frames must be at most
250 ms old and no more than 150 ms apart. The cameras are not hardware synchronized.

The NUC may continuously send observation messages. The bridge always drains
the connection in a small background reader. During inference and chunk
execution, incoming observations are **discarded**, not queued. The whole chunk
is sent immediately when inference returns; the receiver does not gate that send.
For each discarded observation, the bridge sends:

```json
{"status":"busy","request_id":"the-incoming-id","message":"Processing right now; observation discarded."}
```

The NUC must keep reading these informational messages; do not ACK or resend an
observation in response to `busy`. Wait for the next `get_observation` request.

After the completion ACK, the bridge sends a new `get_observation` with a new
`request_id`. The NUC must sample new robot state after that request and echo
its ID. Subsequent stream observations may keep that ID until the next request.
Old IDs are ignored even if their packets arrive late. Do not put a new ID on a
previously sampled state. This prevents old TCP-buffered observations
from being used for the next chunk. Only the first matching reply is accepted
for each request. There is no observation queue or saved frame for a future run.

2. After inference the bridge sends one message:

```python
{
    "type": "action_chunk",
    "chunk_id": "unique-session:0",
    "action_hz": 15,
    "policy_output": {
        "action": [[q1, q2, q3, q4, q5, q6, q7, gripper_score], ...],
        # All other fields returned by Cosmos are preserved here too.
    },
}
```

There are 32 rows. All eight values are passed through unchanged: seven absolute
joint targets in radians and the model's raw gripper score. The NUC applies the
checkpoint's gripper convention (`score > 0.5` means closed) and uses its
existing controller for execution. The bridge does not clip, interpolate,
binarize, or select a subset of the actions.

After finishing the chunk's execution/dispatch interval, the NUC replies:

```python
{"status": "completed", "chunk_id": "unique-session:0"}
```

The bridge then requests a new observation. Add `"done": true` to this receipt,
or to an observation reply, to end the loop. Errors use
`{"status":"error","message":"reason"}`. A queued/accepted receipt is not
completion. Motion commands are never retried automatically.

If the controller ends a task partway through a chunk, it must report the
executed prefix explicitly, for example
`{"status":"stopped","done":true,"chunk_id":"unique-session:0","executed_actions":15}`.
The bridge exits cleanly and does not send another chunk. This is not a full
completion ACK; the remaining rows were cancelled. A `completed` reply with an
explicit count other than 32 is rejected.

The NumPy/MsgPack codec follows NVIDIA RoboLab's Apache-2.0 transport for
compatibility with the official Cosmos server. See `LICENSE`.

## Local ZED cameras

Default roles are wrist Mini **12583732**, external-left 2i **33625030**, and
external-right 2i **36860439**. Override with
`--camera-serials WRIST_SERIAL LEFT_SERIAL RIGHT_SERIAL`. `bridge.py` uses
`cameras.ZedCameras`, whose default capture path is now UVC/V4L2. The local
`uvc_capture.py` contains the restored driver and factory calibration code;
there are no imports from the older bridge project or RoboLab.

The SDK enumerates serial-to-video-device mappings without opening its camera
or MCU/HID interface. Frames are read through accessible `/dev/video*` nodes.
Only the left optical image is used, with Mini fisheye or 2i radial/tangential
factory rectification. USB 3 cameras use 1280x720 per eye at 30 FPS; the USB 2
2i negotiates 672x376 at 15 FPS. This camera rate is separate from action timing.

The SDK at `~/.local/opt/zed-sdk-5.5.1-cu13` is loaded automatically. Set `ZED_DIR`
for a different SDK directory and `ZED_SETTINGS_PATH` for the existing factory
calibration files. Install the updated requirements in this project's own
environment to add `opencv-python-headless` (no GUI preview is needed).
Test cameras without contacting either the policy server or NUC:

```bash
.venv/bin/python cameras.py --list
.venv/bin/python cameras.py --output outputs/zed_capture.png
```

The same commands now use UVC by default. Capture writes the PNG and a matching
JSON sidecar containing camera serials, negotiated modes and driver capture
timestamps. All three cameras must open successfully. Each capture drains old
frames and waits for new exposures after that capture request. UVC capture needs
read/write access to the video devices, but does not require MCU/HID access.
SDK enumeration alone does not verify that images can be captured.

If video-device permissions are also missing, an administrator can grant access
over SSH using the existing helper (it also covers ZED USB/HID nodes):

```bash
sudo python3 /home/sangjunpark/Documents/cosmos-tcp-bridge/setup/grant_zed_access.py --user sangjunpark
```

The helper selects only currently connected Stereolabs video, HID, and USB
device nodes (vendor `2b03`) and grants that user read/write ACLs. Preview with
`--dry-run`. It changes no NUC or policy settings and requires no desktop login.
This grant lasts for the current device connections; rerun it after unplugging
cameras or rebooting, or have the administrator configure persistent device rules.

## Test with Isaac cameras and a RoboLab NUC stand-in

The optional [dummy_robolab](dummy_robolab/README.md) folder runs the actual
bridge against the real local Cosmos server and a loopback-only RoboLab robot.
It substitutes Isaac RGB images for physical ZED captures; robot state and whole
chunks still travel through the normal TCP protocol. Run from this directory:

```bash
.venv/bin/python dummy_robolab/run.py
```

The 2026-10-25 test passed the full banana task in 180 actions / 12 simulation
seconds. Six policy queries produced five full chunks and a final 20-action
prefix. All 262 extra observations received `busy` replies, with no extra model
queries or actions after success. Protocol failure checks passed separately.
This test does not validate physical ZED permissions or real NUC timing.

The new executor can also be tested independently of Isaac, cameras and a live
model, together with the bridge protocol tests:

```bash
.venv/bin/python -m unittest discover -s dummy_robolab -p 'test_*.py' -v
```

`test_uvc_cameras.py` checks role packing, old-frame rejection, fresh exposures
on successive requests, capture deadlines/skew, driver timestamps and cleanup
after a partial open. The SDK camera-open path is explicitly forbidden in that test.

`test_nuc_executor.py` checks the actual executor over loopback TCP, a two-chunk
bridge cycle, fresh measured feedback, wall-clock dispatch timing, gripper
conversion, busy/fragmented/pipelined requests, replay rejection, invalid
inputs, reported faults, slow RPC calls, disconnects and interruption. If
ZeroRPC is importable, it also launches the actual executor CLI and unchanged
DROID `ServerInterface` against a dummy ZeroRPC service bound to `127.0.0.1:4242`.
Set `DROID_DIR` to that checkout root if it differs from `~/Documents/dbs_stuff/droid`.
That optional test proceeds only after its own dummy service binds successfully.
It never contacts the physical NUC. No DROID launch/reset methods are called.
