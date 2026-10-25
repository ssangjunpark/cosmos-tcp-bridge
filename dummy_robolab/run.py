"""Reproducible full test: actual bridge + real Cosmos + loopback RoboLab NUC."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def validate(root):
    sim = root / "simulator"
    result = json.loads((sim / "result.json").read_text())
    audit = json.loads((root / "client_audit.json").read_text())
    requests = json.loads((sim / "requests.json").read_text())
    chunks = json.loads((sim / "chunks.json").read_text())
    steps = [json.loads(line) for line in (sim / "steps.jsonl").read_text().splitlines()]
    calls, frames = audit["model_calls"], audit["camera_frames"]
    assert not result["error"] and not audit["error"], (result["error"], audit["error"])
    assert len(requests) == len(chunks) == len(calls) == len(frames) > 0
    assert all(address[0] == "127.0.0.1" for address in audit["connections"])
    assert result["extra_observations_sent"] == result["busy_replies"] > 0
    consumed = 0
    for i, (request, chunk, call, frame) in enumerate(zip(requests, chunks, calls, frames)):
        with np.load(root / f"input_{i:03d}.npz") as input:
            np.testing.assert_allclose(
                input["observation/joint_position"],
                request["state"]["observation/joint_position"],
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                input["observation/gripper_position"],
                request["state"]["observation/gripper_position"],
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                input["observation/joint_position"], frame["measured_q"], atol=1e-6, rtol=0
            )
        assert frame["reference_image_match"]
        assert frame["sim_step"] == request["sim_step"] == consumed
        assert request["received_at"] <= frame["requested_at"] <= frame["captured_at"] <= call["started_at"]
        assert call["returned_at"] <= chunk["started_at"]
        if i:
            assert request["received_at"] >= chunks[i - 1]["ack_sent_at"]
            np.testing.assert_allclose(
                request["state"]["observation/joint_position"],
                steps[consumed - 1]["measured_state"]["observation/joint_position"],
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                request["state"]["observation/gripper_position"],
                steps[consumed - 1]["measured_state"]["observation/gripper_position"],
                atol=1e-6,
                rtol=0,
            )
        packet = json.loads((sim / f"chunk_{i:03d}.json").read_text())
        output = json.loads((root / f"output_{i:03d}.json").read_text())
        assert packet["policy_output"] == output, "Policy output was modified in transit"
        actions = np.load(root / f"actions_{i:03d}.npy")
        np.testing.assert_array_equal(np.asarray(packet["policy_output"]["action"], np.float32), actions)
        count = chunk["receipt"]["executed_actions"]
        for j in range(count):
            actual = steps[consumed + j]
            assert actual["chunk"] == i and actual["row"] == j
            assert actual["step"] == consumed + j + 1
            np.testing.assert_array_equal(np.asarray(actual["policy_action"], np.float32), actions[j])
            expected = actions[j].copy()
            expected[7] = float(expected[7] > 0.5)
            np.testing.assert_array_equal(np.asarray(actual["applied_action"], np.float32), expected)
        consumed += count
    episode = result["environment_results"][0]
    assert consumed == len(steps) == result["steps"] == episode["step"], "Post-termination actions detected"
    assert chunks[-1]["receipt"]["status"] == "stopped" and chunks[-1]["receipt"]["done"] is True
    return {
        "pipeline_passed": True,
        "task_succeeded": episode["success"],
        "steps": len(steps),
        "simulation_seconds": len(steps) / 15,
        "model_queries": len(calls),
        "busy_replies": result["busy_replies"],
        "extra_observations_discarded": result["extra_observations_sent"],
        "inference_seconds": [call["latency_s"] for call in calls],
        "inference_return_to_nuc_seconds": [
            chunk["started_at"] - call["returned_at"] for chunk, call in zip(chunks, calls)
        ],
        "full_policy_output_unchanged": True,
        "reference_camera_packing_exact": True,
        "all_frames_after_fresh_state": True,
        "measured_feedback_verified": True,
        "no_extra_queries_from_discarded_observations": True,
        "no_actions_after_task_end": True,
        "last_chunk_executed_actions": chunks[-1]["receipt"]["executed_actions"],
        "video": str(sim / "rollout.mp4"),
        "physical_nuc_contacted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robolab-dir", type=Path, default=Path.home() / "Documents/RoboLab")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = (
        args.output_dir or PROJECT / "outputs" / datetime.now().strftime("robolab_%Y%m%d_%H%M%S_%f")
    ).resolve()
    root.mkdir(parents=True, exist_ok=False)
    sim = root / "simulator"
    sim.mkdir()
    latest = PROJECT / "outputs/robolab_latest.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(root) + "\n")
    print("Test output:", root, flush=True)
    with urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("Local Cosmos server is unhealthy")
    with (root / "protocol_tests.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(PROJECT / "dummy_robolab/test_protocol.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    environment = dict(os.environ)
    environment["ROBOLAB_DIR"] = str(args.robolab_dir.resolve())
    compatibility = Path.home() / ".cache/cosmos-robolab-warp-1.8.1"
    if not compatibility.is_dir():
        raise RuntimeError("Install the test-only Warp cache; see dummy_robolab/README.md")
    environment["PYTHONPATH"] = str(compatibility) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    auth = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
    if not environment.get("DISPLAY") and auth.is_file() and auth.stat().st_uid == os.getuid():
        displays = [p for p in Path("/tmp/.X11-unix").glob("X*") if p.stat().st_uid == os.getuid()]
        if len(displays) == 1:
            environment.update(DISPLAY=":" + displays[0].name[1:], XAUTHORITY=str(auth))
    icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if icd.is_file():
        environment.setdefault("VK_ICD_FILENAMES", str(icd))
    command = [
        str(args.robolab_dir / ".venv/bin/python"),
        "-u",
        str(PROJECT / "dummy_robolab/sim_nuc.py"),
        "--output-dir",
        str(sim),
        "--seed",
        str(args.seed),
        "--headless",
        "--enable_cameras",
    ]
    with (root / "simulator.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=args.robolab_dir, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    (root / "simulator.pid").write_text(str(process.pid))
    result = {"status": "starting", "seed": args.seed, "physical_nuc_contacted": False}
    try:
        deadline = time.monotonic() + 180
        while not (sim / "ready.json").exists():
            if process.poll() is not None:
                raise RuntimeError("Simulator exited during setup")
            if time.monotonic() > deadline:
                raise TimeoutError("Simulator setup timed out")
            time.sleep(0.25)
        with (root / "bridge.log").open("w") as log:
            completed = subprocess.run(
                [sys.executable, "-u", str(PROJECT / "dummy_robolab/client.py"), "--output-dir", str(root)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=420,
            )
        result["bridge_returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError("Bridge failed; inspect bridge.log")
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (sim / "stop").touch()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        result["simulator_returncode"] = process.returncode
        (root / "result.json").write_text(json.dumps(result, indent=2))
    try:
        assert process.returncode == 0, "Simulator did not shut down cleanly"
        result.update(validate(root))
        result["status"] = "passed" if result["task_succeeded"] else "task_failed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not result["task_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
