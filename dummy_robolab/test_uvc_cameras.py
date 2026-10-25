"""UVC integration and freshness checks; no camera hardware or robot connection."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cameras  # noqa: E402
import uvc_capture  # noqa: E402


class UvcCameraTests(unittest.TestCase):
    def camera(self, streams):
        camera = cameras.ZedCameras()
        camera.cameras = streams
        x, y = np.meshgrid(np.arange(16, dtype=np.float32), np.arange(8, dtype=np.float32))
        camera.maps = [(x, y)] * 3
        return camera

    @staticmethod
    def stream(color, delay=0, stale_once=False):
        class Stream:
            path = "fake-video"

            def __init__(self):
                self.calls = 0

            def read(self, timeout):
                self.calls += 1
                time.sleep(delay)
                captured = time.monotonic()
                if stale_once and self.calls == 1:
                    return np.zeros((8, 16, 3), np.uint8), captured - 1
                return np.full((8, 16, 3), color, np.uint8), captured

            def close(self):
                pass

        return Stream()

    def test_fresh_rgb_roles_and_no_frame_reuse_on_next_request(self):
        streams = [
            self.stream(color, stale_once=True) for color in ((240, 0, 0), (0, 240, 0), (0, 0, 240))
        ]
        camera = self.camera(streams)
        image = camera.capture()
        self.assertEqual(image.shape, (540, 640, 3))
        np.testing.assert_array_equal(image[180, 320], [240, 0, 0])
        np.testing.assert_array_equal(image[450, 160], [0, 240, 0])
        np.testing.assert_array_equal(image[450, 480], [0, 0, 240])
        first = camera.last_capture["captured_at"]
        requested = time.monotonic()
        camera.capture()
        self.assertTrue(all(t >= requested for t in camera.last_capture["captured_at"].values()))
        self.assertTrue(
            all(t > first[role] for role, t in camera.last_capture["captured_at"].items())
        )
        self.assertEqual([s.calls for s in streams], [3, 3, 3])

    def test_deadline_and_exposure_skew_reject_delayed_frames(self):
        for delay, timeout in ((0.04, 0.01), (0.18, 1)):
            with self.subTest(delay=delay):
                camera = self.camera(
                    [self.stream((1, 2, 3)), self.stream((1, 2, 3)), self.stream((1, 2, 3), delay)]
                )
                with self.assertRaises(TimeoutError):
                    camera.capture(timeout=timeout)
                self.assertIsNone(camera.last_capture)

    def test_stale_forever_times_out(self):
        stream = self.stream((1, 2, 3))
        stream.read = lambda **_: (np.zeros((8, 16, 3), np.uint8), time.monotonic() - 1)
        camera = self.camera([stream] * 3)
        with self.assertRaises(TimeoutError):
            camera.capture(timeout=0.03)

    def test_sdk_is_used_for_enumeration_only_and_partial_open_closes_streams(self):
        instances = []

        class Stream:
            width, height = 2560, 720

            def __init__(self, path):
                self.closed = False
                instances.append(self)

            def describe(self):
                return {}

            def close(self):
                self.closed = True

        class EnumerationOnly:
            def __init__(self):
                raise AssertionError("SDK Camera instances must never be opened")

            @staticmethod
            def get_device_list():
                return []

        devices = {serial: {"path": f"fake:{serial}"} for serial in cameras.DEFAULT_SERIALS}
        with (
            patch.object(
                cameras, "sdk", return_value=(SimpleNamespace(Camera=EnumerationOnly), Path("/tmp"))
            ),
            patch.object(uvc_capture, "discover_devices", return_value=devices),
            patch.object(uvc_capture, "UvcStream", Stream),
            patch.object(
                uvc_capture,
                "left_maps",
                side_effect=[((None, None), "fisheye"), RuntimeError("missing calibration")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing calibration"):
                with cameras.ZedCameras():
                    self.fail("Partial open must fail")
        self.assertEqual(len(instances), 2)
        self.assertTrue(all(s.closed for s in instances))

    def test_driver_timestamp_validation(self):
        self.assertEqual(uvc_capture.capture_time(1.0, 0x12000, 30, 1.01), 1.0)
        self.assertAlmostEqual(uvc_capture.capture_time(1.0, 0x2000, 30, 1.01), 1 - 1 / 30)
        for flags, timestamp in ((0, 1.0), (0x2000, -2.0), (0x2000, 2.0), (0x2000, float("nan"))):
            with self.subTest(flags=flags, timestamp=timestamp):
                with self.assertRaises(RuntimeError):
                    uvc_capture.capture_time(timestamp, flags, 30, 1.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
