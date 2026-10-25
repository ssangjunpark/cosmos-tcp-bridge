"""Failure tests for the actual bridge using a loopback NUC and model/camera stubs."""

import contextlib
import io
import json
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def exercise(self, mode):
        calls, received, busy, errors = [], [], [], []
        inference = threading.Event()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)
        port = listener.getsockname()[1]
        state = {"observation/joint_position": [0.0] * 7, "observation/gripper_position": [0.2]}

        def serve():
            try:
                with listener:
                    connection, _ = listener.accept()
                    with connection, connection.makefile("rb") as stream:

                        def send(message):
                            data = (json.dumps(message) + "\n").encode()
                            if mode == "fragmented":
                                for index in range(0, len(data), 11):
                                    connection.sendall(data[index : index + 11])
                                    time.sleep(0.0005)
                            else:
                                connection.sendall(data)

                        while line := stream.readline():
                            message = json.loads(line)
                            if message.get("status") == "busy":
                                busy.append(message)
                                continue
                            received.append(message)
                            if message["type"] == "get_observation":
                                if mode == "malformed_json":
                                    connection.sendall(b"{broken\n")
                                    continue
                                if mode == "already_done":
                                    send({"request_id": message["request_id"], "done": True})
                                    continue
                                observation = dict(state)
                                if mode == "bad_gripper":
                                    observation["observation/gripper_position"] = [7.0]
                                send({"request_id": message["request_id"], "observation": observation})
                                if mode == "disconnect":
                                    inference.wait(2)
                                    return
                                if mode == "flood":
                                    inference.wait(2)
                                    for _ in range(40):
                                        send(
                                            {
                                                "request_id": message["request_id"],
                                                "observation": {"poison": True},
                                            }
                                        )
                            else:
                                reply = {"status": "completed", "chunk_id": message["chunk_id"], "done": True}
                                if mode == "queued":
                                    reply["status"] = "queued"
                                if mode == "wrong_id":
                                    reply["chunk_id"] = "wrong"
                                if mode == "missing_ack":
                                    continue
                                if mode in ("stopped", "invalid_stop"):
                                    reply.update(
                                        status="stopped", executed_actions=5 if mode == "stopped" else 33
                                    )
                                if mode == "false_completion":
                                    reply["executed_actions"] = 5
                                send(reply)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as exc:
                errors.append(repr(exc))

        class Cameras:
            def __init__(self, *_):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def capture(self):
                if mode == "camera_failure":
                    raise RuntimeError("camera failed")
                return np.zeros((540, 640, 3), np.uint8)

        class Policy:
            def __init__(self):
                self.metadata = True

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def send(self, packet):
                calls.append(msgpack.unpackb(packet, object_hook=bridge.decode_numpy, strict_map_key=False))
                inference.set()

            def recv(self, **_):
                if self.metadata:
                    self.metadata = False
                    return msgpack.packb({})
                if mode in ("flood", "disconnect"):
                    time.sleep(0.08)
                if mode == "bad_policy_text":
                    return "inference failed"
                actions = np.zeros((32, 8), np.float32)
                if mode == "bad_shape":
                    actions = actions[:3]
                if mode == "nan":
                    actions[0, 0] = np.nan
                return msgpack.packb({"action": actions}, default=bridge.encode_numpy)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        failure = None
        with (
            patch.object(bridge, "ZedCameras", Cameras),
            patch.object(bridge, "connect", return_value=Policy()),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    bridge.main(
                        [
                            "--nuc-host",
                            "127.0.0.1",
                            "--nuc-port",
                            str(port),
                            "--prompt",
                            "test",
                            "--timeout",
                            "0.4",
                        ]
                    )
                except Exception as exc:
                    failure = exc
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        return failure, calls, received, busy

    def test_flood_is_discarded_and_replied_busy_without_extra_inference(self):
        failure, calls, received, busy = self.exercise("flood")
        self.assertIsNone(failure)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(busy), 40)
        self.assertEqual([r["type"] for r in received], ["get_observation", "action_chunk"])

    def test_fragmented_tcp_packets_are_reassembled(self):
        failure, calls, _, _ = self.exercise("fragmented")
        self.assertIsNone(failure)
        self.assertEqual(len(calls), 1)

    def test_stopped_terminal_prefix_ends_cleanly(self):
        failure, calls, received, _ = self.exercise("stopped")
        self.assertIsNone(failure)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(received), 2)

    def test_early_done_never_captures_or_queries_policy(self):
        failure, calls, received, _ = self.exercise("already_done")
        self.assertIsNone(failure)
        self.assertFalse(calls)
        self.assertEqual(len(received), 1)

    def test_failures_stop_without_retry_or_another_observation(self):
        for mode in (
            "queued",
            "wrong_id",
            "missing_ack",
            "disconnect",
            "malformed_json",
            "bad_gripper",
            "camera_failure",
            "bad_policy_text",
            "bad_shape",
            "nan",
            "invalid_stop",
            "false_completion",
        ):
            with self.subTest(mode=mode):
                failure, calls, received, _ = self.exercise(mode)
                self.assertIsNotNone(failure, mode)
                self.assertLessEqual(len(calls), 1)
                self.assertLessEqual(sum(r["type"] == "action_chunk" for r in received), 1)
                self.assertEqual(sum(r["type"] == "get_observation" for r in received), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
