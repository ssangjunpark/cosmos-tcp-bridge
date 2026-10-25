"""On-request RGB capture from the three ZEDs connected to the 5090."""

import argparse
import ctypes
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
            image.resize(size, Image.Resampling.BILINEAR), ((640 - size[0]) // 2, (360 - size[1]) // 2)
        )
        return np.asarray(canvas)

    def half(rgb):
        # Match the reference client's non-antialiased 2x bilinear reduction.
        return (rgb.astype(np.uint16).reshape(180, 2, 320, 2, 3).sum(axis=(1, 3)) // 4).astype(np.uint8)

    return np.concatenate([padded(wrist), np.concatenate([half(padded(left)), half(padded(right))], axis=1)])


class ZedCameras:
    def __init__(self, serials=DEFAULT_SERIALS):
        if len(serials) != 3 or len(set(serials)) != 3:
            raise ValueError("Provide different wrist, left and right ZED serials")
        self.serials, self.cameras = tuple(serials), []

    def __enter__(self):
        self.sl, root = sdk()
        try:
            for role, serial in zip(("wrist", "left", "right"), self.serials):
                init = self.sl.InitParameters()
                init.set_from_serial_number(serial)
                init.camera_resolution = self.sl.RESOLUTION.VGA  # Also fits the USB 2 camera.
                init.camera_fps = 15
                init.depth_mode = self.sl.DEPTH_MODE.NONE
                init.sensors_required = False
                init.open_timeout_sec = 5.0
                init.async_grab_camera_recovery = True
                settings = Path(os.environ.get("ZED_SETTINGS_PATH", root / "settings"))
                if settings.is_dir():
                    init.optional_settings_path = str(settings) + "/"
                camera = self.sl.Camera()
                self.cameras.append(camera)
                result = camera.open(init)
                if result != self.sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(
                        f"Cannot open {role} ZED {serial}: {result}; check USB/device permissions"
                    )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def capture(self, timeout=2.0):
        def grab(camera):
            after = camera.get_timestamp(self.sl.TIME_REFERENCE.CURRENT).get_nanoseconds()
            deadline = time.monotonic() + timeout
            image = self.sl.Mat()
            while time.monotonic() < deadline:
                if camera.grab() != self.sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.005)
                    continue
                captured = camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                now = camera.get_timestamp(self.sl.TIME_REFERENCE.CURRENT).get_nanoseconds()
                if captured < after or not 0 <= now - captured <= 250_000_000:
                    continue  # Discard old SDK frames; never relabel or save them for later.
                if camera.retrieve_image(image, self.sl.VIEW.LEFT) != self.sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError("ZED retrieve_image failed")
                return np.ascontiguousarray(image.get_data()[..., :3][..., ::-1])  # BGRA -> RGB.
            raise TimeoutError("ZED did not produce a fresh image after the NUC state arrived")

        with ThreadPoolExecutor(max_workers=3) as workers:
            images = list(workers.map(grab, self.cameras))
        return compose_image(*images)

    def __exit__(self, *_):
        for camera in self.cameras:
            camera.close()
        self.cameras.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--camera-serials", nargs=3, type=int, default=DEFAULT_SERIALS)
    parser.add_argument("--output", type=Path, default=Path("outputs/zed_capture.png"))
    args = parser.parse_args()
    if args.list:
        sl, _ = sdk()
        print("ZED SDK", sl.Camera.get_sdk_version())
        for device in sl.Camera.get_device_list():
            print(device.serial_number, device.camera_model, device.camera_state)
    else:
        with ZedCameras(args.camera_serials) as cameras:
            rgb = cameras.capture()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output)
        print(f"Saved live RGB {rgb.shape} to {args.output}")
