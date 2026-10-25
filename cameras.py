"""On-request UVC RGB capture from the three ZEDs connected to the 5090."""

import argparse
import ctypes
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SERIALS = (12583732, 33625030, 36860439)  # Wrist Mini, left 2i, right 2i.


def sdk():
    local = Path.home() / ".local/opt/zed-sdk-5.5.1-cu13"
    root = Path(os.environ.get("ZED_DIR", local if local.is_dir() else "/usr/local/zed"))
    library = root / "lib/libsl_zed.so"
    if library.is_file():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    import pyzed.sl as sl

    return sl, root


def compose_image(wrist, left, right):
    def padded(rgb):
        image = Image.fromarray(rgb)
        ratio = max(image.width / 640, image.height / 360)
        size = (max(1, int(image.width / ratio)), max(1, int(image.height / ratio)))
        canvas = Image.new("RGB", (640, 360))
        canvas.paste(
            image.resize(size, Image.Resampling.BILINEAR),
            ((640 - size[0]) // 2, (360 - size[1]) // 2),
        )
        return np.asarray(canvas)

    def half(rgb):
        # Match the reference client's non-antialiased 2x bilinear reduction.
        return (rgb.astype(np.uint16).reshape(180, 2, 320, 2, 3).sum(axis=(1, 3)) // 4).astype(
            np.uint8
        )

    return np.concatenate(
        [padded(wrist), np.concatenate([half(padded(left)), half(padded(right))], axis=1)]
    )


class ZedCameras:
    def __init__(self, serials=DEFAULT_SERIALS):
        if len(serials) != 3 or len(set(serials)) != 3:
            raise ValueError("Provide different wrist, left and right ZED serials")
        self.serials, self.cameras = tuple(serials), []
        self.maps, self.details, self.last_capture = [], {}, None

    def __enter__(self):
        from uvc_capture import UvcStream, discover_devices, left_maps

        sl, root = sdk()  # Enumeration only; never call the SDK's Camera.open().
        devices = discover_devices(sl.Camera.get_device_list())
        settings = Path(os.environ.get("ZED_SETTINGS_PATH", root / "settings"))
        try:
            for role, serial in zip(("wrist", "left", "right"), self.serials):
                if serial not in devices:
                    raise RuntimeError(
                        f"No UVC video interface for {role} ZED {serial}; found {list(devices)}"
                    )
                stream = UvcStream(devices[serial]["path"])
                self.cameras.append(stream)
                maps, model = left_maps(serial, (stream.width // 2, stream.height), settings)
                self.maps.append(maps)
                self.details[role] = {
                    **devices[serial],
                    **stream.describe(),
                    "lens_model": model,
                    "rectification": "factory",
                }
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def capture(self, timeout=2.0):
        import cv2

        if len(self.cameras) != 3:
            raise RuntimeError("Open all three UVC cameras before capture")
        requested = time.monotonic()
        deadline = requested + timeout
        self.last_capture = None

        def grab(pair):
            stream, maps = pair
            while time.monotonic() < deadline:
                try:
                    rgb, captured = stream.read(
                        timeout=min(0.2, max(0, deadline - time.monotonic()))
                    )
                except TimeoutError:
                    continue
                if captured < requested:
                    continue  # Drain old driver buffers before accepting a new exposure.
                rectified = cv2.remap(rgb, *maps, interpolation=cv2.INTER_LINEAR)
                return rectified, captured
            raise TimeoutError(f"No fresh UVC frame from {stream.path} after the capture request")

        with ThreadPoolExecutor(max_workers=3) as workers:
            frames = list(workers.map(grab, zip(self.cameras, self.maps)))
        now = time.monotonic()
        times = [frame[1] for frame in frames]
        if now > deadline or any(not 0 <= now - timestamp <= 0.25 for timestamp in times):
            raise TimeoutError("UVC capture exceeded its deadline or returned stale frames")
        if max(times) - min(times) > 0.15:
            raise TimeoutError("UVC camera exposures are more than 150 ms apart")
        for role, stream in zip(("wrist", "left", "right"), self.cameras):
            if role in self.details:
                self.details[role].update(stream.describe())
        self.last_capture = {
            "backend": "uvc",
            "requested_at": requested,
            "captured_at": dict(zip(("wrist", "left", "right"), times)),
            "frame_age_s": dict(zip(("wrist", "left", "right"), [now - t for t in times])),
            "capture_skew_s": max(times) - min(times),
            "cameras": self.details,
        }
        return compose_image(*(frame[0] for frame in frames))

    def __exit__(self, *_):
        for camera in self.cameras:
            camera.close()
        self.cameras.clear()
        self.maps.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--camera-serials", nargs=3, type=int, default=DEFAULT_SERIALS)
    parser.add_argument("--output", type=Path, default=Path("outputs/zed_capture.png"))
    args = parser.parse_args()
    if args.list:
        from uvc_capture import discover_devices

        sl, _ = sdk()
        print(json.dumps(discover_devices(sl.Camera.get_device_list()), indent=2))
    else:
        with ZedCameras(args.camera_serials) as cameras:
            rgb = cameras.capture()
            metadata = cameras.last_capture
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output)
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved live RGB {rgb.shape} to {args.output}")
