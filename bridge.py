# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# NumPy/MsgPack encoding follows NVIDIA RoboLab's Apache-2.0 transport.
"""NUC state + local ZED images -> official Cosmos -> whole action chunk to NUC."""

import argparse
import json
import socket
import threading
import time
import uuid

import msgpack
import numpy as np
from websockets.sync.client import connect

from cameras import DEFAULT_SERIALS, ZedCameras


def encode_numpy(value):
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value)}")


def decode_numpy(value):
    if b"__ndarray__" in value:
        return np.frombuffer(value[b"data"], np.dtype(value[b"dtype"])).reshape(value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def json_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value)}")


class NucStream:
    """Hand off one requested reply; reject other observations without queueing."""

    def __init__(self, connection, timeout):
        self.connection, self.timeout = connection, timeout
        self.changed = threading.Condition()
        self.write_lock = threading.Lock()
        self.expected = self.reply = self.error = None
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        try:
            with self.connection.makefile("rb") as stream:
                while True:
                    line = stream.readline(8 * 1024 * 1024 + 1)
                    if not line.endswith(b"\n") or len(line) > 8 * 1024 * 1024:
                        raise ConnectionError("Missing or oversized NUC message")
                    message = json.loads(line)
                    if not isinstance(message, dict) or message.get("status") == "error":
                        raise RuntimeError(f"NUC error: {message}")
                    with self.changed:
                        accepted = self.expected and message.get(self.expected[0]) == self.expected[1]
                        if accepted:
                            self.reply = message
                            self.expected = None  # Close admission immediately; no latest-frame slot.
                            self.changed.notify()
                    if not accepted and ("observation" in message or message.get("type") == "observation"):
                        self.send(
                            {
                                "status": "busy",
                                "request_id": message.get("request_id"),
                                "message": "Processing right now; observation discarded.",
                            }
                        )
        except Exception as exc:
            with self.changed:
                self.error = exc
                self.changed.notify()

    def send(self, message):
        data = (json.dumps(message, default=json_numpy, allow_nan=False) + "\n").encode()
        with self.write_lock:
            self.connection.sendall(data)

    def exchange(self, message, key, value):
        with self.changed:
            self.expected, self.reply = (key, value), None
        try:
            if self.error:
                raise self.error
            self.send(message)
            with self.changed:
                ready = self.changed.wait_for(lambda: self.reply is not None or self.error, self.timeout)
                if self.reply is not None:
                    return self.reply
                if self.error:
                    raise self.error
                if not ready:
                    raise TimeoutError("NUC did not send the requested observation/completion")
        finally:
            with self.changed:
                self.expected = self.reply = None

    def close(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.reader.join(timeout=2)


def observation_request(observation, image, prompt, session):
    """Combine local camera RGB with the measured joint/gripper state from NUC."""
    if image.shape != (540, 640, 3) or image.dtype != np.uint8:
        raise ValueError("Expected local ZED RGB uint8 [540,640,3]")
    joints = np.asarray(observation["observation/joint_position"], dtype=np.float32)
    gripper = np.asarray(observation["observation/gripper_position"], dtype=np.float32)
    if joints.shape != (7,) or gripper.shape != (1,) or not np.isfinite(joints).all():
        raise ValueError("Expected seven finite joint positions and one gripper value")
    if not np.isfinite(gripper).all() or not 0 <= gripper[0] <= 1:
        raise ValueError("NUC gripper state must be normalized: 0=open, 1=closed")
    return {
        "observation/image": image,
        "observation/joint_position": joints,
        "observation/gripper_position": gripper,
        "prompt": prompt,
        "session_id": session,
    }


def query_policy(policy, request, timeout):
    policy.send(msgpack.packb(request, default=encode_numpy))
    raw = policy.recv(timeout=timeout)
    if isinstance(raw, str):
        raise RuntimeError(f"Cosmos error: {raw}")
    response = msgpack.unpackb(raw, object_hook=decode_numpy, strict_map_key=False)
    if not isinstance(response, dict) or "action" not in response:
        raise RuntimeError(f"Cosmos did not return an action chunk: {response}")
    actions = np.asarray(response["action"])
    if actions.shape != (32, 8) or not np.isfinite(actions).all():
        raise ValueError("Expected a finite [32,8] action chunk")
    return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuc-host", required=True)
    parser.add_argument("--nuc-port", type=int, default=9001)
    parser.add_argument("--policy-uri", default="ws://127.0.0.1:8000/")
    parser.add_argument(
        "--camera-serials", nargs=3, type=int, default=DEFAULT_SERIALS, metavar=("WRIST", "LEFT", "RIGHT")
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--chunks", type=int, default=0, help="Chunks to run; 0 (default) repeats until stopped"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.chunks < 0 or args.timeout <= 0:
        parser.error("Use a nonnegative chunk count and a positive timeout")
    session = uuid.uuid4().hex
    with (
        ZedCameras(args.camera_serials) as cameras,
        connect(args.policy_uri, compression=None, max_size=None, open_timeout=args.timeout) as policy,
    ):
        metadata = policy.recv(timeout=args.timeout)  # Official server sends metadata first.
        if isinstance(metadata, str):
            raise RuntimeError(f"Cosmos handshake error: {metadata}")
        metadata = msgpack.unpackb(metadata, object_hook=decode_numpy, strict_map_key=False)
        if isinstance(metadata, dict) and metadata.get("type") == "error":
            raise RuntimeError(f"Cosmos handshake error: {metadata}")
        with socket.create_connection((args.nuc_host, args.nuc_port), timeout=args.timeout) as connection:
            nuc = NucStream(connection, args.timeout)
            try:
                index = 0
                while args.chunks == 0 or index < args.chunks:
                    observation_id = f"{session}:observation:{index}"
                    reply = nuc.exchange(
                        {"type": "get_observation", "request_id": observation_id},
                        "request_id",
                        observation_id,
                    )
                    if reply.get("done") is True:
                        break
                    image = cameras.capture()
                    request = observation_request(reply["observation"], image, args.prompt, session)
                    started = time.monotonic()
                    output = query_policy(policy, request, args.timeout)
                    inference_time = time.monotonic() - started
                    chunk_id = f"{session}:{index}"
                    receipt = nuc.exchange(
                        {
                            "type": "action_chunk",
                            "chunk_id": chunk_id,
                            "action_hz": 15,
                            "policy_output": output,
                        },
                        "chunk_id",
                        chunk_id,
                    )
                    executed = receipt.get("executed_actions", 32)
                    completed = (
                        receipt.get("status") == "completed" and type(executed) is int and executed == 32
                    )
                    stopped = (
                        receipt.get("status") == "stopped"
                        and receipt.get("done") is True
                        and type(receipt.get("executed_actions")) is int
                        and 0 <= executed <= 32
                    )
                    if receipt.get("chunk_id") != chunk_id or not (completed or stopped):
                        raise RuntimeError(f"NUC did not confirm chunk completion: {receipt}")
                    print(
                        f"chunk={index} inference={inference_time:.3f}s {receipt['status']} actions={executed}/32",
                        flush=True,
                    )
                    index += 1
                    if receipt.get("done") is True:
                        break
            finally:
                nuc.close()


if __name__ == "__main__":
    main()
