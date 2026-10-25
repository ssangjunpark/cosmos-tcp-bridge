"""Loopback NUC/controller/robot using RoboLab physics and Isaac RGB cameras."""

import argparse
import json
import os
import select
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ROBOLAB_DIR", Path.home() / "Documents/RoboLab"))
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(REPO)]
import warp as wp  # noqa: E402 -- load the R580-compatible Warp before Isaac.

wp.init()
from isaaclab.app import AppLauncher  # noqa: E402


class Wire:
    """Only TCP framing bytes are buffered; commands are processed immediately."""

    def __init__(self, connection, root):
        self.connection, self.root, self.buffer = connection, root, b""
        self.busy = 0

    def send(self, message):
        self.connection.sendall((json.dumps(message, allow_nan=False) + "\n").encode())

    def receive(self, timeout=0):
        if b"\n" not in self.buffer:
            if not select.select([self.connection], [], [], timeout)[0]:
                return None
            data = self.connection.recv(65536)
            if not data:
                raise EOFError("Bridge closed TCP")
            self.buffer += data
            if len(self.buffer) > 8 * 1024 * 1024:
                raise ValueError("Oversized message")
            if b"\n" not in self.buffer:
                return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        message = json.loads(line)
        if message.get("status") == "busy":
            self.busy += 1
            with (self.root / "busy.jsonl").open("a") as out:
                out.write(json.dumps(message) + "\n")
        return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="BananaInBowlTask")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    env = writer = listener = connection = None
    steps, requests, chunks, extras = 0, [], [], 0
    failure = None
    try:
        import torch
        from isaaclab.sensors import Camera
        from policies.cosmos3.client import Cosmos3Client
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.task.instruction import primary_instruction
        from robolab.core.utils.video_utils import VideoWriter
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT

        from cameras import compose_image

        set_output_dir(str(root / "scene"))
        auto_register_droid_envs(task=[args.task], cameras=WRIST_LEFT_RIGHT)
        (name,) = get_envs(task=[args.task])
        env, cfg = create_env(
            name,
            seed=args.seed,
            num_envs=1,
            device=args.device,
            policy="minimal-tcp-bridge",
            instruction_type=["default"],
            renderer="realtime",
            rendering_mode="performance",
        )
        obs, _ = env.reset(seed=args.seed)
        dt = float(env.step_dt)
        if not np.isclose(dt, 1 / 15):
            raise ValueError(f"Unexpected simulator action period: {dt}")
        robot = env.scene["robot"]
        joints = [robot.data.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        finger = robot.data.joint_names.index("finger_joint")
        sensors = [s for s in env.scene.sensors.values() if isinstance(s, Camera)]
        writer = VideoWriter(str(root / "rollout.mp4"), 15)
        reference = Cosmos3Client.__new__(Cosmos3Client)  # Packing only; opens no policy connection.
        reference._image_h, reference._image_w, reference._eval_episode_idx = 360, 640, 0

        def measured():
            q = robot.data.joint_pos[0, joints].detach().cpu().numpy().copy()
            g = float(np.clip(float(robot.data.joint_pos[0, finger].cpu()) / (np.pi / 4), 0, 1))
            return {"observation/joint_position": q.tolist(), "observation/gripper_position": [g]}

        def images(observation):
            return {
                role: observation["image_obs"][name][0].detach().cpu().numpy()[..., :3].copy()
                for role, name in [
                    ("wrist", "wrist_cam"),
                    ("left", "over_shoulder_left_camera"),
                    ("right", "over_shoulder_right_camera"),
                ]
            }

        def render():
            captured = time.monotonic()
            env.sim.render()
            for sensor in sensors:
                sensor.update(max(dt, float(sensor.cfg.update_period)), force_recompute=True)
            observation = env.observation_manager.compute(update_history=False)
            expected = reference._pack_request(reference._extract_observation(observation), "test")[
                "observation/image"
            ]
            return captured, images(observation), expected

        for _ in range(5):
            _, initial, _ = render()
        writer.write(compose_image(**initial))
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(120)
        _, prompt = primary_instruction(cfg.instruction)
        ready = {
            "host": "127.0.0.1",
            "port": listener.getsockname()[1],
            "prompt": prompt,
            "max_steps": int(env.max_episode_length),
            "seed": args.seed,
            "physical_nuc_contacted": False,
        }
        (root / "ready.json").write_text(json.dumps(ready, indent=2))
        print("SIMULATED NUC READY " + json.dumps(ready), flush=True)
        connection, _ = listener.accept()
        connection.settimeout(60)
        wire = Wire(connection, root)
        last_request = capture_id = None
        last_extra = time.monotonic()
        while app.is_running() and not (root / "stop").exists():
            capture_file = root / "capture_request.json"
            if capture_file.exists():
                capture = json.loads(capture_file.read_text())
                if capture["id"] != capture_id:
                    captured, rgb, expected = render()
                    temporary = root / "frames.tmp.npz"
                    np.savez(
                        temporary,
                        **rgb,
                        capture_id=capture["id"],
                        captured_at=captured,
                        sim_step=steps,
                        measured_q=measured()["observation/joint_position"],
                        reference_image=expected,
                    )
                    temporary.replace(root / "frames.npz")
                    capture_id = capture["id"]
            if last_request and not env.all_terminated and time.monotonic() - last_extra > 0.05:
                wire.send({"request_id": last_request, "observation": measured()})
                extras += 1
                last_extra = time.monotonic()
            message = wire.receive(timeout=0.01)
            if not message or message.get("status") == "busy":
                continue
            if message["type"] == "get_observation":
                if last_request:
                    wire.send(
                        {"request_id": last_request, "observation": {"old_packet_must_be_discarded": True}}
                    )
                    extras += 1
                last_request = message["request_id"]
                state = measured()
                requests.append(
                    {
                        "request_id": last_request,
                        "sim_step": steps,
                        "received_at": time.monotonic(),
                        "state": state,
                    }
                )
                wire.send(
                    {"request_id": last_request, "observation": state, "done": bool(env.all_terminated)}
                )
                (root / "requests.json").write_text(json.dumps(requests, indent=2))
                continue
            if message["type"] != "action_chunk" or env.all_terminated:
                raise ValueError("Unexpected command or command after episode termination")
            actions = np.asarray(message["policy_output"]["action"], np.float32)
            if actions.shape != (32, 8) or not np.isfinite(actions).all() or message["action_hz"] != 15:
                raise ValueError("Invalid whole chunk")
            index = len(chunks)
            (root / f"chunk_{index:03d}.json").write_text(json.dumps(message))
            started, count = time.monotonic(), 0
            for row_index, row in enumerate(actions):
                if env.all_terminated:
                    raise RuntimeError("Attempted post-termination physics step")
                applied = row.copy()
                applied[7] = float(row[7] > 0.5)
                obs, _, _, _, _ = env.step(torch.tensor(applied, device=env.device).unsqueeze(0))
                steps += 1
                count += 1
                state = measured()
                writer.write(compose_image(**images(obs)))
                record = {
                    "step": steps,
                    "chunk": index,
                    "row": row_index,
                    "policy_action": row.tolist(),
                    "applied_action": applied.tolist(),
                    "measured_state": state,
                }
                with (root / "steps.jsonl").open("a") as out:
                    out.write(json.dumps(record) + "\n")
                wire.send({"request_id": last_request, "observation": state})
                extras += 1
                while incoming := wire.receive():
                    if incoming.get("status") != "busy":
                        raise RuntimeError("Bridge sent another command during chunk execution")
                if env.all_terminated:
                    break
            terminal = bool(env.all_terminated)
            receipt = {
                "status": "stopped" if terminal else "completed",
                "chunk_id": message["chunk_id"],
                "executed_actions": count,
                "done": terminal,
            }
            wire.send(receipt)
            chunks.append(
                {
                    "chunk_id": message["chunk_id"],
                    "started_at": started,
                    "ack_sent_at": time.monotonic(),
                    "receipt": receipt,
                }
            )
            (root / "chunks.json").write_text(json.dumps(chunks, indent=2))
            print(f"SIM CHUNK {index}: {count}/32 steps={steps} terminal={terminal}", flush=True)
    except EOFError:
        pass  # Normal: bridge closes the connection after the terminal receipt.
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            result = {
                "error": failure,
                "steps": steps,
                "simulated_seconds": steps / 15,
                "environment_results": env.get_env_results(),
                "chunks": len(chunks),
                "extra_observations_sent": extras,
                "busy_replies": wire.busy if connection else 0,
                "physical_nuc_contacted": False,
            }
            (root / "result.json").write_text(json.dumps(result, indent=2))
            print("SIM RESULT " + json.dumps(result), flush=True)
        for resource in (connection, listener):
            if resource is not None:
                resource.close()
        if writer is not None:
            writer.release()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
