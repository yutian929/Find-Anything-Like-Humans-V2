"""当前发布端协议和相机几何约束的回归测试。"""

import unittest

import msgpack
import numpy as np

from frame_source import decode_rgbd_pose_message


class DecodeRGBDPoseMessageTest(unittest.TestCase):
    def setUp(self):
        self.meta = {
            "version": 2,
            "type": "rgbd_pose",
            "frame_id": 7,
            "timestamp_ns": 1_700_000_000_000_000_000,
            "width": 2,
            "height": 1,
            "depth_width": 2,
            "depth_height": 1,
            "rgb_dtype": "uint8",
            "depth_dtype": "uint16",
            "rgb_encoding": "rgb8",
            "depth_encoding": "z16",
            "depth_aligned_to": "color",
            "T_map_camera_frame": "color_optical",
            "depth_frame": "color_optical",
            "depth_scale_m": 0.001,
            "pose_valid": True,
            "T_map_camera": np.eye(4).tolist(),
            "color_intr": {"fx": 1, "fy": 1, "ppx": 0, "ppy": 0},
        }

    def decode(self):
        return decode_rgbd_pose_message([
            b"rgbd.pose",
            msgpack.packb(self.meta, use_bin_type=True),
            bytes([1, 2, 3, 4, 5, 6]),
            np.array([1000, 2000], dtype=np.uint16).tobytes(),
        ])

    def test_v2_aligned_frame(self):
        frame = self.decode()
        self.assertEqual(frame.rgb.tolist(), [[[3, 2, 1], [6, 5, 4]]])
        self.assertEqual(frame.depth.tolist(), [[1000, 2000]])
        self.assertEqual(frame.frame_id, 7)

    def test_v1_is_rejected(self):
        self.meta["version"] = 1
        with self.assertRaisesRegex(ValueError, "协议版本"):
            self.decode()

    def test_v2_rejects_unaligned_depth(self):
        self.meta["depth_aligned_to"] = None
        with self.assertRaisesRegex(ValueError, "深度对齐"):
            self.decode()


if __name__ == "__main__":
    unittest.main()
