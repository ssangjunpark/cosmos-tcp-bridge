# SPDX-License-Identifier: Apache-2.0
"""Run on the NUC: bridge TCP messages -> existing local DROID robot API."""

import argparse
import json
import select
import socket
import sys
import time
from pathlib import Path

import numpy as np


class Wire:
    """Frame TCP bytes; requests received while busy are never executed later."""

    def __init__(self, connection, wait=select.select):
        self.connection, self.wait = connection, wait
        self.buffer, self.busy_bytes = b"", False

    def send(self, message):
        self.connection.sendall((json.dumps(message, allow_nan=False) + "\n").encode())

    def receive(self, timeout, busy=False):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            if not self.wait([self.connection], [], [], max(0, deadline - time.monotonic()))[0]:
                return None
            data = self.connection.recv(65536)
            if not data:
                raise EOFError("Bridge disconnected")
            self.busy_bytes = self.busy_bytes or busy
            self.buffer += data
            if self.buffer.find(b"\n") < 0 and len(self.buffer) > 8 * 1024 * 1024:
                raise ValueError("Oversized TCP message")
        line, self.buffer = self.buffer.split(b"\n", 1)
        if len(line) > 8 * 1024 * 1024:
            raise ValueError("Oversized TCP message")
        discarded = self.busy_bytes or busy
        self.busy_bytes = discarded if self.buffer else False
        message = json.loads(line)
        if not isinstance(message, dict):
            raise TypeError("Expected a JSON object")
        return message, discarded

    def reject(self, message):
        if message.get("status") != "busy":
            self.send(
                {
                    "status": "busy",
                    "request_id": message.get("request_id"),
                    "chunk_id": message.get("chunk_id"),
                    "message": "Executing a chunk; request discarded.",
                }
            )

    def wait_busy(self, deadline):
        while True:
            received = self.receive(max(0, deadline - time.monotonic()), busy=True)
            if received is None:
                return
            self.reject(received[0])


def measured_state(robot):
    state, timestamp = robot.get_robot_state()
    joints = np.asarray(state["joint_positions"], dtype=float)
    gripper = float(state["gripper_position"])
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("DROID must return seven finite measured joint positions")
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("DROID gripper position must be 0=open through 1=closed")
    return {
        "observation/joint_position": joints.tolist(),
        "observation/gripper_position": [gripper],
    }, timestamp


def serve_connection(connection, robot, timeout=60.0, wait=select.select):
    """One bridge session. Completion counts returned update_command calls."""
    wire = Wire(connection, wait)
    awaiting_chunk, seen_chunks = False, set()
    message, chunk_id, dispatched = {}, None, 0
    try:
        while True:
            received = wire.receive(timeout)
            if received is None:
                raise TimeoutError("Timed out waiting for the bridge")
            message, discarded = received
            if discarded:
                wire.reject(message)
                continue
            if message.get("status") == "busy":
                continue  # Informational; never acknowledge or retry a busy reply.
            chunk_id, dispatched = None, 0
            if message.get("type") == "get_observation":
                request_id = message.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("get_observation requires a nonempty request_id")
                observation, timestamp = measured_state(robot)
                wire.send(
                    {
                        "status": "ok",
                        "request_id": request_id,
                        "observation": observation,
                        "timestamp": timestamp,
                    }
                )
                awaiting_chunk = True
                continue
            if message.get("type") != "action_chunk":
                raise ValueError("Expected get_observation or action_chunk")
            chunk_id = message.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen_chunks:
                raise ValueError("Expected a new, nonempty chunk_id; chunks are never replayed")
            if not awaiting_chunk:
                raise ValueError("Request fresh measured state before sending a chunk")
            if message.get("action_hz") != 15:
                raise ValueError("DROID policy actions must use action_hz=15")
            actions = np.asarray(message["policy_output"]["action"], dtype=float)
            if actions.shape != (32, 8) or not np.isfinite(actions).all():
                raise ValueError("Expected a finite whole action chunk [32,8]")
            seen_chunks.add(chunk_id)
            awaiting_chunk = False
            wire.busy_bytes = bool(wire.buffer)  # Also discard already pipelined requests.
            started, period = time.monotonic(), 1 / 15
            for action in actions:
                wire.wait_busy(
                    time.monotonic()
                )  # Detect disconnect / discard requests before dispatch.
                tick = time.monotonic()
                command = action.copy()
                command[7] = float(action[7] > 0.5)  # DROID checkpoint: 1=closed, 0=open.
                robot.update_command(
                    command,
                    action_space="joint_position",
                    gripper_action_space="position",
                    blocking=False,
                )
                dispatched += 1
                if time.monotonic() - tick >= period:
                    raise TimeoutError("DROID command submission exceeded the 15 Hz action period")
                wire.wait_busy(tick + period)
            wire.send(
                {
                    "status": "completed",
                    "chunk_id": chunk_id,
                    "executed_actions": dispatched,
                    "completion_basis": "dispatched",
                    "elapsed_s": time.monotonic() - started,
                }
            )
            print(
                f"chunk={chunk_id} dispatched={dispatched}/32; awaiting fresh state request",
                flush=True,
            )
            chunk_id = None
    except EOFError:
        if chunk_id is not None:
            raise  # Stop dispatching immediately; never continue a disconnected session.
    except KeyboardInterrupt:
        if chunk_id is not None:
            wire.send(
                {
                    "status": "stopped",
                    "done": True,
                    "chunk_id": chunk_id,
                    "executed_actions": dispatched,
                    "completion_basis": "dispatched",
                }
            )
        raise
    except Exception as exc:
        try:
            wire.send(
                {
                    "status": "error",
                    "message": str(exc),
                    "request_id": message.get("request_id"),
                    "chunk_id": chunk_id,
                    "executed_actions": dispatched,
                    "completion_basis": "dispatched",
                }
            )
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="NUC address to listen on")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument(
        "--droid-dir", type=Path, help="DROID checkout root, if not installed in this environment"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Bridge TCP idle timeout in seconds"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Use a valid TCP port and a finite positive timeout")
    if args.droid_dir:
        sys.path.insert(0, str(args.droid_dir.expanduser().resolve()))

    # Use DROID's existing environment. Cooperative sockets keep ZeroRPC heartbeats running.
    from droid.misc.server_interface import ServerInterface
    from gevent import select as cooperative_select
    from gevent import socket as cooperative_socket

    robot = ServerInterface(ip_address="127.0.0.1", launch=False)
    try:
        measured_state(robot)  # Read-only readiness check; never launch, reset or home the robot.
        with cooperative_socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            print(
                f"NUC executor listening on {args.host}:{args.port}; DROID=127.0.0.1:4242",
                flush=True,
            )
            connection, peer = listener.accept()
            with connection:
                connection.settimeout(args.timeout)
                print(f"Bridge connected: {peer}", flush=True)
                serve_connection(connection, robot, args.timeout, cooperative_select.select)
    finally:
        robot.server.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Executor stopped; no further targets dispatched.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary; report failures and exit, never retry motion.
        print(f"Executor stopped: {exc}", file=sys.stderr)
        sys.exit(1)
