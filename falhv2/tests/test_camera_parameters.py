import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from falhv2.tools import camera_parameters as camera
def profile(stream="color", image_format="bgr8", fps=30, fx=100):
    result = MagicMock()
    result.stream_type.return_value = "stream."+stream
    result.format.return_value = "format."+image_format
    result.width.return_value = 200
    result.height.return_value = 100
    result.fps.return_value = fps
    result.stream_index.return_value = 0
    result.is_video_stream_profile.return_value = True
    result.as_video_stream_profile.return_value = result
    result.get_intrinsics.return_value = SimpleNamespace(
        width=200, height=100, fx=fx, fy=100, ppx=100, ppy=50,
        model="distortion.inverse_brown_conrady", coeffs=[.01, -.02, .003, .004, .005])
    return result


def device(serial="test-serial"):
    result = MagicMock()
    info = {"name": "RealSense D435I", "serial_number": serial,
            "firmware_version": "test-firmware", "product_line": "D400", "usb_type_descriptor": "3.2"}
    result.supports.return_value = True
    result.get_info.side_effect = info.__getitem__
    sensor = MagicMock()
    sensor.get_stream_profiles.return_value = [profile(), profile("depth", "z16", fx=80)]
    result.query_sensors.return_value = [sensor]
    return result


def sdk(devices):
    result = MagicMock()
    result.stream = SimpleNamespace(color="stream.color", depth="stream.depth")
    result.camera_info = SimpleNamespace(**{key: key for key in
        ("name", "serial_number", "firmware_version", "product_line", "usb_type_descriptor")})
    result.context.return_value.query_devices.return_value = devices
    return result


class CameraParameterTests(unittest.TestCase):
    def test_sdk_intrinsics_and_distortion_are_preserved_and_manual_fields_marked(self):
        result = camera.make_camera_config(profile())
        intr = result["camera"]
        self.assertEqual([intr[key] for key in ("width", "height", "fx", "fy", "cx", "cy")],
                         [200, 100, 100, 100, 100, 50])
        self.assertEqual(intr["distortion_coefficients"], [.01, -.02, .003, .004, .005])
        self.assertEqual(intr["distortion_model"], "inverse_brown_conrady")
        self.assertEqual(set(result), {"camera", "mount", "observation_quality"})
        self.assertEqual(result["mount"]["position_in_base_m"], [None]*3)
        self.assertTrue(all(result["mount"][key] is None for key in ("roll_deg", "pitch_deg", "yaw_deg")))
        self.assertTrue(all(value is None for value in result["observation_quality"].values()))
        self.assertEqual(set(result["observation_quality"]), {"min_distance_m", "max_distance_m"})
        for invalid in (0, -1, math.nan, math.inf):
            with self.assertRaises(camera.CameraReadError):
                camera.make_camera_config(profile(fx=invalid))

    def test_mode_selection_matches_resolution_format_and_fps_exactly(self):
        p30, p60 = profile(), profile(fps=60)
        self.assertIs(camera.select_profile([p30, p60], 200, 100, 60, "BGR8"), p60)
        for args in ((200, 100, 15, "BGR8"), (640, 480, 30, "BGR8"), (200, 100, 30, "Z16")):
            with self.assertRaises(camera.CameraReadError):
                camera.select_profile([p30, p60], *args)
        with self.assertRaises(camera.CameraReadError):
            camera.select_profile([p30, p30], 200, 100, 30, "BGR8")

    def test_multicamera_selection_and_color_depth_separation(self):
        d1, d2 = device(), device("second")
        rs = sdk([d1, d2])
        with self.assertRaises(camera.CameraReadError):
            camera.select_device([d1, d2], rs, None)
        self.assertIs(camera.select_device([d1, d2], rs, "second"), d2)
        for devices, serial in (([], None), ([d1], "missing")):
            with self.assertRaises(camera.CameraReadError):
                camera.select_device(devices, rs, serial)
        rgb = camera.video_profiles(d1, rs, "color")
        depth = camera.video_profiles(d1, rs, "depth")
        self.assertEqual(len(rgb), 1)
        self.assertEqual(camera.make_camera_config(rgb[0])["camera"]["fx"], 100)
        self.assertEqual(camera.make_camera_config(depth[0])["camera"]["fx"], 80)

    def test_cli_generates_template_and_leaves_user_default_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"outputs/camera.json"
            default = Path(folder)/"data/camera.json"
            default.parent.mkdir()
            default.write_text('{"manually_measured_mount": true}\n')
            with patch.object(camera, "load_sdk", return_value=sdk([device()])), contextlib.redirect_stdout(io.StringIO()):
                args = ["--width", "200", "--height", "100", "--output", str(output)]
                self.assertEqual(camera.main(args, camera_path=default), 0)
                saved = json.loads(output.read_text())
                self.assertEqual(set(saved["camera"]), {
                    "width", "height", "fx", "fy", "cx", "cy",
                    "distortion_model", "distortion_coefficients"})
                self.assertIsNone(saved["mount"]["yaw_deg"])
                self.assertTrue(json.loads(default.read_text())["manually_measured_mount"])
                before = output.read_bytes()
                self.assertEqual(camera.main(args+["--list-modes"], camera_path=default), 0)
                self.assertEqual(output.read_bytes(), before)



if __name__ == "__main__":
    unittest.main()
