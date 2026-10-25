"""Run the actual bridge, replacing only ZED hardware with Isaac's rendered images."""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import bridge  # noqa: E402
from cameras import compose_image  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    sim = root / "simulator"
    ready = json.loads((sim / "ready.json").read_text())
    if ready["host"] != "127.0.0.1":
        raise ValueError("Test controller must be literal loopback")
    frames, calls, exchanges, connections = [], [], [], []

    class IsaacCameras:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def capture(self):
            identifier = f"capture-{len(frames)}"
            requested = time.monotonic()
            temporary = sim / "capture_request.tmp"
            temporary.write_text(json.dumps({"id": identifier}))
            temporary.replace(sim / "capture_request.json")
            deadline = requested + 15
            while time.monotonic() < deadline:
                path = sim / "frames.npz"
                if path.exists():
                    with np.load(path) as data:
                        if str(data["capture_id"]) == identifier:
                            captured = float(data["captured_at"])
                            if captured < requested:
                                raise ValueError("Isaac image predates capture request")
                            image = compose_image(data["wrist"], data["left"], data["right"])
                            if not np.array_equal(image, data["reference_image"]):
                                raise ValueError("Camera packing differs from RoboLab's reference client")
                            np.savez_compressed(
                                root / f"camera_{len(frames):03d}.npz",
                                **{name: data[name] for name in ("wrist", "left", "right")},
                            )
                            frames.append(
                                {
                                    "requested_at": requested,
                                    "captured_at": captured,
                                    "sim_step": int(data["sim_step"]),
                                    "measured_q": data["measured_q"].tolist(),
                                    "reference_image_match": True,
                                }
                            )
                            return image
                time.sleep(0.005)
            raise TimeoutError("Isaac camera capture timed out")

    original_query = bridge.query_policy
    original_exchange = bridge.NucStream.exchange
    original_connection = socket.create_connection

    def connect(address, *a, **kw):
        if address[0] != "127.0.0.1":
            raise ValueError("Test cannot connect to physical hardware")
        connections.append(list(address))
        return original_connection(address, *a, **kw)

    def query(policy, request, timeout):
        index = len(calls)
        np.savez_compressed(
            root / f"input_{index:03d}.npz",
            **{k: v for k, v in request.items() if k.startswith("observation/")},
        )
        started = time.monotonic()
        output = original_query(policy, request, timeout)
        returned = time.monotonic()
        np.save(root / f"actions_{index:03d}.npy", output["action"])
        (root / f"output_{index:03d}.json").write_text(json.dumps(output, default=bridge.json_numpy))
        calls.append({"started_at": started, "returned_at": returned, "latency_s": returned - started})
        return output

    def exchange(self, message, *a, **kw):
        sent = time.monotonic()
        reply = original_exchange(self, message, *a, **kw)
        exchanges.append(
            {"type": message["type"], "sent_at": sent, "received_at": time.monotonic(), "reply": reply}
        )
        return reply

    bridge.ZedCameras = IsaacCameras
    bridge.query_policy = query
    bridge.NucStream.exchange = exchange
    socket.create_connection = connect
    failure = None
    try:
        bridge.main(
            [
                "--nuc-host",
                "127.0.0.1",
                "--nuc-port",
                str(ready["port"]),
                "--prompt",
                ready["prompt"],
                "--timeout",
                "60",
            ]
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (root / "client_audit.json").write_text(
            json.dumps(
                {
                    "error": failure,
                    "camera_frames": frames,
                    "model_calls": calls,
                    "exchanges": exchanges,
                    "connections": connections,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
