"""Exercise the actual NUC executor over loopback TCP; never contact hardware."""

import contextlib
import importlib.util
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge  # noqa: E402
import nuc_executor  # noqa: E402


class Robot:
    def __init__(self, fail_at=None, slow=False, interrupt=False):
        self.joints = np.array([0.01, -0.2, 0.03, -1.8, 0.05, 1.5, 0.07])
        self.gripper = 0.2
        self.calls, self.reads = [], 0
        self.fail_at, self.slow, self.interrupt = fail_at, slow, interrupt

    def get_robot_state(self):
        self.reads += 1
        return {"joint_positions": self.joints.tolist(), "gripper_position": self.gripper}, {
            "robot_timestamp_seconds": self.reads,
            "robot_timestamp_nanos": 123,
        }

    def update_command(self, command, **kwargs):
        if len(self.calls) == self.fail_at:
            if self.interrupt:
                raise KeyboardInterrupt
            raise RuntimeError("dummy controller submission failed")
        self.calls.append((command.copy(), kwargs, time.monotonic()))
        self.joints = command[:7] + 0.001  # Deliberate tracking difference: feedback != target.
        self.gripper = float(np.clip(command[7], 0, 1))
        if self.slow:
            time.sleep(0.08)


class NucExecutorTests(unittest.TestCase):
    @contextlib.contextmanager
    def session(self, robot=None, timeout=4):
        robot = robot or Robot()
        errors = []
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)

        def serve():
            try:
                with listener:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(4)
                        nuc_executor.serve_connection(connection, robot, timeout)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        with socket.create_connection(listener.getsockname(), timeout=4) as client:
            wire = nuc_executor.Wire(client)
            try:
                yield robot, client, wire, errors
            finally:
                with contextlib.suppress(OSError):
                    client.shutdown(socket.SHUT_RDWR)
        worker.join(3)
        self.assertFalse(worker.is_alive(), "Executor did not stop after disconnect")

    def receive(self, wire):
        received = wire.receive(4)
        self.assertIsNotNone(received)
        return received[0]

    def state(self, wire, request_id="state:0"):
        wire.send({"type": "get_observation", "request_id": request_id})
        reply = self.receive(wire)
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["request_id"], request_id)
        return reply

    @staticmethod
    def chunk(chunk_id="chunk:0"):
        actions = np.tile(Robot().joints.tolist() + [0.2], (32, 1))
        actions[:, 0] += np.arange(32) * 0.001
        actions[:, 7] = np.linspace(-0.1, 1.1, 32)
        return {
            "type": "action_chunk",
            "chunk_id": chunk_id,
            "action_hz": 15,
            "policy_output": {"action": actions.tolist(), "server_timing": {"infer": 0.42}},
        }

    def test_real_bridge_two_chunks_fresh_feedback_timing_and_hold(self):
        model_requests, captures = [], []
        chunks = [self.chunk(f"chunk:{i}") for i in range(2)]

        class Cameras:
            def __init__(self, *_):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def capture(self):
                captures.append(time.monotonic())
                return np.zeros((540, 640, 3), np.uint8)

        class Policy:
            def __init__(self):
                self.metadata = True

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def send(self, packet):
                model_requests.append(
                    msgpack.unpackb(packet, object_hook=bridge.decode_numpy, strict_map_key=False)
                )

            def recv(self, **_):
                if self.metadata:
                    self.metadata = False
                    return msgpack.packb({})
                before = len(robot.calls)
                time.sleep(0.1)  # During inference no update_command / synthetic hold is issued.
                if len(robot.calls) != before:
                    raise AssertionError("Robot received a command during inference")
                return msgpack.packb(chunks[len(model_requests) - 1]["policy_output"])

        with self.session() as (robot, client, _, errors):
            # The session's accepted socket is the actual bridge's persistent TCP connection.
            with (
                patch.object(bridge.socket, "create_connection", return_value=client),
                patch.object(bridge, "ZedCameras", Cameras),
                patch.object(bridge, "connect", return_value=Policy()),
            ):
                bridge.main(
                    [
                        "--nuc-host",
                        "127.0.0.1",
                        "--prompt",
                        "test",
                        "--chunks",
                        "2",
                        "--timeout",
                        "4",
                    ]
                )
        self.assertFalse(errors, errors)
        self.assertEqual(len(captures), 2)
        self.assertEqual(robot.reads, 2)
        self.assertEqual(len(robot.calls), 64)
        expected = np.vstack([c["policy_output"]["action"] for c in chunks])
        dispatched = expected.copy()
        dispatched[:, 7] = (expected[:, 7] > 0.5).astype(float)
        np.testing.assert_array_equal([c[0] for c in robot.calls], dispatched)
        np.testing.assert_allclose(
            model_requests[1]["observation/joint_position"], expected[31, :7] + 0.001, atol=1e-7
        )
        self.assertEqual(float(model_requests[1]["observation/gripper_position"][0]), 1)
        for _, kwargs, _ in robot.calls:
            self.assertEqual(
                kwargs,
                {
                    "action_space": "joint_position",
                    "gripper_action_space": "position",
                    "blocking": False,
                },
            )
        for start in (0, 32):
            ticks = [c[2] for c in robot.calls[start : start + 32]]
            self.assertGreaterEqual(min(np.diff(ticks)), 1 / 15 - 0.005)
            self.assertLess(max(np.diff(ticks)), 0.2)
            self.assertGreaterEqual(ticks[-1] - ticks[0], 31 / 15 - 0.01)

    def test_busy_pipelining_partial_frame_and_replay_are_discarded(self):
        with self.session() as (robot, client, wire, errors):
            self.state(wire)
            payload = (
                json.dumps(self.chunk()) + "\n" + json.dumps(self.chunk("queued")) + "\n"
            ).encode()
            for offset in range(0, len(payload), 37):
                client.sendall(payload[offset : offset + 37])
            busy = self.receive(wire)
            self.assertEqual((busy["status"], busy["chunk_id"]), ("busy", "queued"))
            time.sleep(0.1)
            client.sendall(b'{"type":"get_observation","request_id":"partial"')
            receipt = self.receive(wire)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["executed_actions"], 32)
            self.assertEqual(receipt["completion_basis"], "dispatched")
            self.assertGreaterEqual(receipt["elapsed_s"], 32 / 15 - 0.01)
            client.sendall(b"}\n")  # Finish the busy-period frame only after completion.
            self.assertEqual(self.receive(wire)["status"], "busy")
            self.assertEqual(robot.reads, 1)
            self.state(wire, "fresh")
            wire.send(self.chunk())  # A completed chunk ID must never execute twice.
            self.assertEqual(self.receive(wire)["status"], "error")
        self.assertEqual(len(robot.calls), 32)
        self.assertTrue(errors)

    def test_invalid_chunks_never_dispatch_any_action(self):
        for mode in (
            "no_state",
            "shape",
            "nan",
            "hz",
            "missing_id",
            "missing_output",
            "unknown",
            "json",
        ):
            with self.subTest(mode=mode):
                with self.session() as (robot, client, wire, errors):
                    if mode != "no_state":
                        self.state(wire)
                    packet = self.chunk()
                    if mode == "shape":
                        packet["policy_output"]["action"].pop()
                    elif mode == "nan":
                        packet["policy_output"]["action"][0][0] = float("nan")
                    elif mode == "hz":
                        packet["action_hz"] = 30
                    elif mode == "missing_id":
                        del packet["chunk_id"]
                    elif mode == "missing_output":
                        del packet["policy_output"]
                    elif mode == "unknown":
                        packet["type"] = "launch_controller"
                    if mode == "json":
                        client.sendall(b"{broken\n")
                    else:
                        client.sendall((json.dumps(packet) + "\n").encode())
                    self.assertEqual(self.receive(wire)["status"], "error")
                self.assertFalse(robot.calls)
                self.assertTrue(errors)

    def test_invalid_measured_state_stops_before_any_action(self):
        for gripper in (-0.1, 1.1, float("nan")):
            with self.subTest(gripper=gripper):
                robot = Robot()
                robot.gripper = gripper
                with self.session(robot) as (_, _, wire, errors):
                    wire.send({"type": "get_observation", "request_id": "bad_state"})
                    self.assertEqual(self.receive(wire)["status"], "error")
                self.assertFalse(robot.calls)
                self.assertTrue(errors)

    def test_controller_fault_or_slow_submission_stops_without_retry(self):
        for robot, expected in ((Robot(fail_at=3), 3), (Robot(slow=True), 1)):
            with self.subTest(expected=expected):
                with self.session(robot) as (_, _, wire, errors):
                    self.state(wire)
                    wire.send(self.chunk())
                    reply = self.receive(wire)
                    self.assertEqual(reply["status"], "error")
                    self.assertEqual(reply["executed_actions"], expected)
                self.assertEqual(len(robot.calls), expected)
                self.assertTrue(errors)

    def test_disconnect_during_chunk_stops_dispatching(self):
        with self.session() as (robot, client, wire, errors):
            self.state(wire)
            wire.send(self.chunk())
            time.sleep(0.18)
            client.shutdown(socket.SHUT_RDWR)
            time.sleep(0.12)
            count = len(robot.calls)
            time.sleep(0.12)
            self.assertEqual(len(robot.calls), count)
        self.assertGreater(count, 0)
        self.assertLess(count, 32)
        self.assertTrue(errors)

    def test_interrupt_reports_terminal_prefix(self):
        robot = Robot(fail_at=3, interrupt=True)
        with self.session(robot) as (_, _, wire, errors):
            self.state(wire)
            wire.send(self.chunk())
            reply = self.receive(wire)
            self.assertEqual(
                (reply["status"], reply["done"], reply["executed_actions"]), ("stopped", True, 3)
            )
        self.assertEqual(len(robot.calls), 3)
        self.assertIsInstance(errors[0], KeyboardInterrupt)

    @unittest.skipUnless(
        importlib.util.find_spec("zerorpc"), "Optional local ZeroRPC test dependency"
    )
    def test_cli_with_actual_droid_wrapper_and_loopback_zerorpc_server(self):
        project = Path(__file__).resolve().parents[1]
        droid = Path(os.environ.get("DROID_DIR", Path.home() / "Documents/dbs_stuff/droid"))
        self.assertTrue((droid / "droid/misc/server_interface.py").is_file())
        program = """
import json, sys
import numpy as np
import zerorpc
from test_nuc_executor import Robot
class Recorder(Robot):
    def get_robot_state(self):
        state = super().get_robot_state()
        with open(sys.argv[1], "a") as out:
            out.write(json.dumps({"type": "state"}) + "\\n")
        return state
    def update_command(self, command, action_space, gripper_action_space, blocking):
        kwargs = dict(action_space=action_space, gripper_action_space=gripper_action_space, blocking=blocking)
        super().update_command(np.array(command), **kwargs)
        with open(sys.argv[1], "a") as out:
            out.write(json.dumps({"type": "action", "command": command, **kwargs}) + "\\n")
server = zerorpc.Server(Recorder())
server.bind("tcp://127.0.0.1:4242")
print("DUMMY_RPC_READY", flush=True)
server.run()
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(project / "dummy_robolab"), environment.get("PYTHONPATH", "")]
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "rpc.jsonl"
            processes = []
            try:
                rpc = subprocess.Popen(
                    [sys.executable, "-u", "-c", program, str(log)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
                processes.append(rpc)
                self.assertTrue(
                    select.select([rpc.stdout], [], [], 5)[0], "Dummy RPC did not start"
                )
                self.assertEqual(rpc.stdout.readline().strip(), "DUMMY_RPC_READY")
                # Only proceed after our own dummy service successfully owns loopback :4242.
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                executor = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(project / "nuc_executor.py"),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--droid-dir",
                        str(droid),
                        "--timeout",
                        "4",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
                processes.append(executor)
                self.assertTrue(
                    select.select([executor.stdout], [], [], 5)[0], "Executor did not start"
                )
                self.assertIn(f"listening on 127.0.0.1:{port}", executor.stdout.readline())
                with socket.create_connection(("127.0.0.1", port), timeout=4) as client:
                    wire = nuc_executor.Wire(client)
                    self.state(wire)
                    packet = self.chunk()
                    wire.send(packet)
                    receipt = self.receive(wire)
                    self.assertEqual(
                        (receipt["status"], receipt["executed_actions"]), ("completed", 32)
                    )
                stdout, _ = executor.communicate(timeout=5)
                self.assertEqual(executor.returncode, 0, stdout)
                records = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(
                    sum(r["type"] == "state" for r in records), 2
                )  # Startup + request.
                actions = [r for r in records if r["type"] == "action"]
                self.assertEqual(len(actions), 32)
                expected = np.array(packet["policy_output"]["action"])
                expected[:, 7] = expected[:, 7] > 0.5
                np.testing.assert_array_equal([r["command"] for r in actions], expected)
                for record in actions:
                    self.assertEqual(record["action_space"], "joint_position")
                    self.assertEqual(record["gripper_action_space"], "position")
                    self.assertIs(record["blocking"], False)
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
