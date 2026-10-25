# SPDX-License-Identifier: Apache-2.0
"""Linux V4L2 MMAP capture with driver timestamps; never opens a robot connection."""

import configparser
import ctypes as ct
import errno
import fcntl
import mmap
import os
import select
import struct
import time
from pathlib import Path

import cv2
import numpy as np


class Timeval(ct.Structure):
    _fields_ = [("seconds", ct.c_long), ("microseconds", ct.c_long)]


class Timecode(ct.Structure):
    _fields_ = [("type", ct.c_uint32), ("flags", ct.c_uint32), ("values", ct.c_uint8 * 8)]


class BufferMemory(ct.Union):
    _fields_ = [
        ("offset", ct.c_uint32),
        ("userptr", ct.c_ulong),
        ("planes", ct.c_void_p),
        ("fd", ct.c_int),
    ]


class Buffer(ct.Structure):
    _fields_ = [
        ("index", ct.c_uint32),
        ("type", ct.c_uint32),
        ("bytesused", ct.c_uint32),
        ("flags", ct.c_uint32),
        ("field", ct.c_uint32),
        ("timestamp", Timeval),
        ("timecode", Timecode),
        ("sequence", ct.c_uint32),
        ("memory", ct.c_uint32),
        ("m", BufferMemory),
        ("length", ct.c_uint32),
        ("reserved2", ct.c_uint32),
        ("request_fd", ct.c_int32),
    ]


def _ioctl(fd, request, value):
    while True:
        try:
            return fcntl.ioctl(fd, request, value, True)
        except InterruptedError:
            continue


def capture_time(timestamp, flags, fps, now):
    """Require the driver monotonic clock; EOF gets a one-frame earlier bound."""
    if flags & 0xE000 != 0x2000:
        raise RuntimeError("Camera driver does not supply monotonic capture timestamps")
    source = flags & 0x70000
    if source not in (0, 0x10000):
        raise RuntimeError("Unsupported camera timestamp source")
    if not np.isfinite([timestamp, fps]).all() or fps <= 0 or not 0 <= now - timestamp <= 2.0:
        raise RuntimeError("Camera driver timestamp is stale or invalid")
    return timestamp if source == 0x10000 else timestamp - 1.0 / fps


class UvcStream:
    """One nonblocking YUYV camera. The owning thread calls read() continuously."""

    def __init__(self, path, *, fps=30):
        if ct.sizeof(Buffer) != 88:
            raise RuntimeError("This capture implementation requires the Linux 64-bit V4L2 ABI")
        self.path = str(path)
        self.fd = None
        self.buffers = []
        self.streaming = False
        self.timestamp_source = None
        self.last_sequence = None
        try:
            self.fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)
            capability = bytearray(104)
            _ioctl(self.fd, 0x80685600, capability)  # VIDIOC_QUERYCAP
            caps, device_caps = struct.unpack_from("II", capability, 84)
            caps = device_caps if caps & 0x80000000 else caps
            if not caps & 1 or not caps & 0x04000000:
                raise RuntimeError(f"{path} is not a streaming video capture device")
            fourcc = int.from_bytes(b"YUYV", "little")
            modes = []
            for index in range(64):
                size = bytearray(44)
                struct.pack_into("II", size, 0, index, fourcc)
                try:
                    _ioctl(self.fd, 0xC02C564A, size)
                except OSError as exc:
                    if exc.errno == errno.EINVAL:
                        break
                    raise
                kind, width, height = struct.unpack_from("III", size, 8)
                if kind == 1:
                    modes.append((width, height))
            selected = next((mode for mode in ((2560, 720), (1344, 376)) if mode in modes), None)
            if selected is None:
                raise RuntimeError(f"No supported HD720/VGA stereo YUYV mode on {path}: {modes}")
            fmt = bytearray(208)
            struct.pack_into("I", fmt, 0, 1)
            struct.pack_into("IIII", fmt, 8, *selected, fourcc, 0)
            _ioctl(self.fd, 0xC0D05605, fmt)  # VIDIOC_S_FMT
            self.width, self.height, actual_fourcc, _, self.stride = struct.unpack_from(
                "IIIII", fmt, 8
            )
            if (
                (self.width, self.height) != selected
                or actual_fourcc != fourcc
                or self.stride < self.width * 2
            ):
                raise RuntimeError("Video driver returned an unexpected image layout")
            parm = bytearray(204)
            struct.pack_into("I", parm, 0, 1)
            struct.pack_into("II", parm, 12, 1, int(fps))
            _ioctl(self.fd, 0xC0CC5616, parm)  # VIDIOC_S_PARM
            numerator, denominator = struct.unpack_from("II", parm, 12)
            if not numerator or not denominator:
                raise RuntimeError("Camera did not report its frame interval")
            self.fps = denominator / numerator
            request = bytearray(20)
            struct.pack_into("III", request, 0, 4, 1, 1)
            _ioctl(self.fd, 0xC0145608, request)  # VIDIOC_REQBUFS
            count = struct.unpack_from("I", request)[0]
            if not 2 <= count <= 32:
                raise RuntimeError("Unexpected camera buffer count")
            for index in range(count):
                buffer = Buffer(index=index, type=1, memory=1)
                _ioctl(self.fd, 0xC0585609, buffer)  # VIDIOC_QUERYBUF
                self.buffers.append(
                    mmap.mmap(
                        self.fd,
                        buffer.length,
                        mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                        offset=buffer.m.offset,
                    )
                )
                _ioctl(self.fd, 0xC058560F, buffer)  # VIDIOC_QBUF
            _ioctl(self.fd, 0x40045612, ct.c_int(1))  # VIDIOC_STREAMON
            self.streaming = True
        except BaseException:
            self.close()
            raise

    def read(self, timeout=0.2):
        if self.fd is None:
            raise RuntimeError("Camera is closed")
        if not select.select([self.fd], [], [], timeout)[0]:
            raise TimeoutError(f"No video frame from {self.path}")
        latest = None
        for _ in range(len(self.buffers)):
            buffer = Buffer(type=1, memory=1)
            try:
                _ioctl(self.fd, 0xC0585611, buffer)  # VIDIOC_DQBUF
            except BlockingIOError:
                break
            try:
                if buffer.index >= len(self.buffers):
                    raise RuntimeError("Driver returned an invalid buffer index")
                if buffer.flags & 0x40:
                    continue  # driver marks this image corrupt; never publish it
                if buffer.bytesused < self.stride * self.height:
                    raise RuntimeError("Camera returned an incomplete image buffer")
                timestamp = buffer.timestamp.seconds + buffer.timestamp.microseconds / 1e6
                captured = capture_time(timestamp, buffer.flags, self.fps, time.monotonic())
                if self.last_sequence is not None and buffer.sequence == self.last_sequence:
                    continue
                rows = np.frombuffer(
                    self.buffers[buffer.index], dtype=np.uint8, count=self.height * self.stride
                ).reshape(self.height, self.stride)
                # Own the image bytes before requeueing the driver's buffer.
                latest = (rows[:, : self.width * 2].copy(), captured)
                self.last_sequence = int(buffer.sequence)
                self.timestamp_source = (
                    "start_of_exposure"
                    if buffer.flags & 0x70000 == 0x10000
                    else "end_of_frame_minus_one_interval"
                )
            finally:
                _ioctl(self.fd, 0xC058560F, buffer)
        if latest is None:
            raise TimeoutError(f"No valid new frame from {self.path}")
        rows, captured = latest
        left_yuyv = rows.reshape(self.height, self.width, 2)[:, : self.width // 2]
        return cv2.cvtColor(left_yuyv, cv2.COLOR_YUV2RGB_YUY2), captured

    def describe(self):
        return {
            "device": self.path,
            "image_width": self.width // 2,
            "image_height": self.height,
            "fps": self.fps,
            "timestamp_source": self.timestamp_source,
            "timestamp_clock": "monotonic",
            "pixel_format": "YUYV",
        }

    def close(self):
        if self.fd is not None and self.streaming:
            try:
                _ioctl(self.fd, 0x40045613, ct.c_int(1))
            except OSError:
                pass
            self.streaming = False
        for buffer in self.buffers:
            buffer.close()
        self.buffers.clear()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def discover_devices(sdk_devices):
    """Use SDK serial-to-path mapping, plus a same-hub HID match for USB2 ZED2i."""
    sdk = {
        str(d.path): int(d.serial_number)
        for d in sdk_devices
        if int(d.serial_number) > 0 and str(d.path)
    }
    found = {}
    for node in Path("/sys/class/video4linux").glob("video*"):
        try:
            if (node / "index").read_text().strip() != "0":
                continue
            usb = next(
                parent
                for parent in (node / "device").resolve().parents
                if (parent / "idVendor").exists()
            )
            if (usb / "idVendor").read_text().strip() != "2b03":
                continue
            path = str(Path("/dev") / node.name)
            serial = sdk.get(path)
            match = "sdk_device_path"
            if serial is None:
                hub = usb.parent
                if (
                    not (hub / "idVendor").exists()
                    or (hub / "bDeviceClass").read_text().strip() != "09"
                ):
                    continue
                hid = []
                for sibling in hub.iterdir():
                    if sibling == usb or not (sibling / "idProduct").exists():
                        continue
                    if (sibling / "idVendor").read_text().strip() == "2b03" and (
                        sibling / "idProduct"
                    ).read_text().strip() == "f881":
                        hid.append(int((sibling / "serial").read_text().strip()))
                if len(hid) != 1:
                    continue
                serial, match = hid[0], "same_camera_hub_hid_serial"
            if serial in found:
                raise RuntimeError(f"Ambiguous video devices for ZED serial {serial}")
            found[serial] = {
                "serial": serial,
                "path": path,
                "usb_speed_mbps": int((usb / "speed").read_text()),
                "usb_device": usb.name,
                "serial_match": match,
            }
        except (OSError, StopIteration, ValueError):
            continue
    return found


def left_maps(serial, size, settings_dir):
    names = {(1280, 720): "HD", (672, 376): "VGA"}
    if tuple(size) not in names:
        raise ValueError(f"Unsupported calibrated image size {size}")
    name = names[tuple(size)]
    path = Path(settings_dir) / f"SN{serial}.conf"
    config = configparser.ConfigParser()
    if not config.read(path):
        raise RuntimeError(f"Missing factory calibration: {path}")
    matrices, distortions, models = [], [], []
    for eye in ("LEFT", "RIGHT"):
        values = config[f"{eye}_CAM_{name}"]

        def f(key):
            return float(values[key])

        matrices.append(
            np.array([[f("fx"), 0.0, f("cx")], [0.0, f("fy"), f("cy")], [0.0, 0.0, 1.0]])
        )
        if {"p1", "p2"} <= values.keys():
            models.append("radial_tangential")
            distortions.append(
                np.array([f(k) for k in ("k1", "k2", "p1", "p2", "k3")]).reshape(5, 1)
            )
        elif {"k1", "k2", "k3", "k4"} <= values.keys():
            models.append("fisheye")
            distortions.append(np.array([f(k) for k in ("k1", "k2", "k3", "k4")]).reshape(4, 1))
        else:
            raise RuntimeError(f"Unrecognized factory lens model: {path}")
    if models[0] != models[1]:
        raise RuntimeError("Camera eyes have different lens models")
    stereo = config["STEREO"]
    rotation, _ = cv2.Rodrigues(
        np.array([float(stereo[f"{axis}_{name}"]) for axis in ("RX", "CV", "RZ")])
    )
    translation = np.array(
        [
            -float(stereo["Baseline"]),
            float(stereo.get(f"TY_{name}", stereo.get("TY", "0"))),
            float(stereo.get(f"TZ_{name}", stereo.get("TZ", "0"))),
        ]
    ).reshape(3, 1)
    if not all(np.isfinite(x).all() for x in matrices + distortions + [rotation, translation]):
        raise ValueError("Factory calibration contains nonfinite values")
    if models[0] == "fisheye":
        rect_left, _, projection, *_ = cv2.fisheye.stereoRectify(
            matrices[0],
            distortions[0],
            matrices[1],
            distortions[1],
            tuple(size),
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0,
            fov_scale=1.0,
        )
        builder = cv2.fisheye.initUndistortRectifyMap
    else:
        rect_left, _, projection, *_ = cv2.stereoRectify(
            matrices[0],
            distortions[0],
            matrices[1],
            distortions[1],
            tuple(size),
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )
        builder = cv2.initUndistortRectifyMap
    maps = builder(matrices[0], distortions[0], rect_left, projection, tuple(size), cv2.CV_32FC1)
    if not all(np.isfinite(value).all() for value in maps):
        raise ValueError("Factory rectification produced invalid pixel coordinates")
    return maps, models[0]
