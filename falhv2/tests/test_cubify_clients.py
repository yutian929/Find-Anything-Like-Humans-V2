import base64
import itertools
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from falhv2.robot_adapters.base import CameraIntrinsics, RGBDFrame
from falhv2.model_clients import build_cubify_anything
from falhv2.model_clients.cubify_anything import (
    gravity_rotation, make_result, threshold, validate_frame)
from falhv2.model_clients.cubify_anything import (
    decode_frame, decode_result, encode_frame, encode_result)
from falhv2.map_maintainers.cubify import cuboids_to_map_aabbs
from falhv2.robot_adapters.anchor_map import from_anchor_map_frame


def sample_frame(width=16, height=12):
    T = np.array([[0., 0., 1., 10.], [-1., 0., 0., 20.],
                  [0., -1., 0., 1.], [0., 0., 0., 1.]])
    return RGBDFrame("frame-7", 123456789, "unix", "map-a", "rev-2",
                     "color_optical", CameraIntrinsics(width, height, 12., 13., 7., 5.),
                     np.full((height, width, 3), [1, 20, 240], dtype=np.uint8),
                     np.full((height, width), 2.5, dtype=np.float32), T)


def sample_corners():
    return np.array(list(itertools.product((-2., 2.), (-.5, .5), (2., 4.))))


class CubifyGeometryTests(unittest.TestCase):
    def test_legacy_bgr_depth_scale_and_owned_arrays(self):
        frame = sample_frame()
        legacy = SimpleNamespace(
            rgb=frame.rgb[..., ::-1].copy(), depth=np.full((12,16),2500,np.uint16),
            camera_pose=frame.T_map_camera.copy(), stamp=1.25, frame_id=7,
            intrinsics=dict(fx=12., fy=13., cx=7., cy=5., depth_scale=.001))
        converted = from_anchor_map_frame(legacy, map_id="map-a", map_revision="rev-2")
        np.testing.assert_array_equal(converted.rgb, frame.rgb)
        np.testing.assert_allclose(converted.depth_m, 2.5)
        self.assertEqual(converted.timestamp_ns, 1250000000)
        self.assertEqual(converted.clock_domain, "unix")
        legacy.rgb[:] = 0
        legacy.camera_pose[:] = 0
        self.assertEqual(int(converted.rgb[0,0,2]), 240)
        np.testing.assert_array_equal(converted.T_map_camera, frame.T_map_camera)
        legacy.intrinsics["depth_scale"] = -1
        with self.assertRaises(ValueError):
            from_anchor_map_frame(legacy, map_id="map-a", map_revision="rev-2")

    def test_validation_and_invalid_depth(self):
        frame = sample_frame()
        depth = frame.depth_m.copy()
        depth[0,:3] = [np.nan, np.inf, -1]
        _, clean = validate_frame(replace(frame, depth_m=depth))
        np.testing.assert_array_equal(clean[0,:3], 0)
        for bad in [replace(frame, depth_m=frame.depth_m.astype(np.uint16)),
                    replace(frame, depth_m=np.zeros_like(depth)),
                    replace(frame, rgb=frame.rgb.astype(float)),
                    replace(frame, intrinsics=replace(frame.intrinsics, fx=0)),
                    replace(frame, T_map_camera=np.zeros((4,4))),
                    replace(frame, timestamp_ns=0),
                    replace(frame, map_revision="")]:
            with self.subTest(bad=bad.frame_id), self.assertRaises(ValueError):
                validate_frame(bad)

    def test_gravity_ignores_map_yaw_and_tracks_pitch(self):
        frame = sample_frame()
        np.testing.assert_allclose(gravity_rotation(frame), np.eye(3), atol=1e-7)
        a = .61
        yaw = np.array([[np.cos(a),-np.sin(a),0],
                        [np.sin(a),np.cos(a),0],[0,0,1.]])
        T = frame.T_map_camera.copy()
        T[:3,:3] = yaw @ T[:3,:3]
        np.testing.assert_allclose(gravity_rotation(replace(frame,T_map_camera=T)),
                                   np.eye(3), atol=1e-7)
        pitch = np.array([[1.,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        T[:3,:3] = frame.T_map_camera[:3,:3] @ pitch
        actual = gravity_rotation(replace(frame,T_map_camera=T))
        np.testing.assert_allclose(actual @ [0,1,0], T[:3,:3].T @ [0,0,-1], atol=1e-7)
        with self.assertRaises(ValueError):
            gravity_rotation(replace(frame,T_map_camera=np.eye(4)))

    def test_score_filter_and_malformed_outputs(self):
        frame, corners = sample_frame(), sample_corners()
        result = make_result(frame,[.1,.9,.5],[corners]*3,"test-version",.5)
        self.assertEqual([b.confidence for b in result.cuboids], [.9,.5])
        self.assertEqual(make_result(frame,[],[],"test",.5).cuboids, ())
        for scores, boxes in [([2.],[corners]), ([.5],np.zeros((1,8,3))),
                              ([.5],np.full((1,8,3),np.nan)), ([.5],[])]:
            with self.assertRaises(ValueError):
                make_result(frame,scores,boxes,"test",.2)
        for score in [-1, 2, float("nan"), True]:
            with self.assertRaises(ValueError):
                threshold(score)

    def test_map_aabb_transforms_all_corners(self):
        frame = sample_frame()
        a = np.pi/4
        yaw = np.array([[np.cos(a),-np.sin(a),0],
                        [np.sin(a),np.cos(a),0],[0,0,1.]])
        T = frame.T_map_camera.copy()
        T[:3,:3] = yaw @ T[:3,:3]
        frame = replace(frame,T_map_camera=T)
        result = make_result(frame,[.8],[sample_corners()],"test",.25)
        box, = cuboids_to_map_aabbs(frame, result)
        extent = 3/np.sqrt(2)
        center = np.array([10+3/np.sqrt(2),20+3/np.sqrt(2),1])
        np.testing.assert_allclose(box.lower,center-[extent,extent,.5])
        np.testing.assert_allclose(box.upper,center+[extent,extent,.5])
        self.assertEqual(box.map_revision, "rev-2")
        with self.assertRaises(ValueError):
            cuboids_to_map_aabbs(replace(frame,frame_id="other"), result)

    def test_wire_lossless_rgb_metric_depth_and_metadata(self):
        frame = sample_frame()
        frame.depth_m[0,0] = np.nan
        recovered, score = decode_frame(encode_frame(frame,.3))
        np.testing.assert_array_equal(recovered.rgb, frame.rgb)
        np.testing.assert_array_equal(recovered.depth_m, np.nan_to_num(frame.depth_m))
        np.testing.assert_array_equal(recovered.T_map_camera, frame.T_map_camera)
        self.assertEqual(recovered.timestamp_ns, frame.timestamp_ns)
        self.assertEqual(score, .3)
        payload = encode_frame(sample_frame(),.3)
        payload["frame"]["depth_f32_le"] = base64.b64encode(b"bad").decode()
        with self.assertRaises(ValueError):
            decode_frame(payload)

    def test_reject_mismatched_remote_response(self):
        frame = sample_frame()
        payload = encode_result(make_result(frame,[.8],[sample_corners()],"test",0.))
        decoded = decode_result(frame,payload,.5)
        self.assertEqual(len(decoded.cuboids),1)
        for field, wrong in [("frame_id","wrong"),("timestamp_ns",1),
                             ("camera_frame","wrong"),("protocol_version",99),
                             ("model_id","different")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode_result(frame,{**payload,field:wrong},.5)

    def test_factory_and_missing_weight_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_cubify_anything({"mode":"typo"})
        with self.assertRaises(FileNotFoundError):
            build_cubify_anything({"mode":"local","path":"/nonexistent/cutr_rgbd.pth"})
        with self.assertRaises(ValueError):
            build_cubify_anything({"mode":"remote","server_url":"not-a-url"})


class CubifyHTTPTests(unittest.TestCase):
    def test_real_http_roundtrip_empty_error_and_close(self):
        from werkzeug.serving import make_server
        from falhv2.tools.cubify_server import create_app

        class Detector:
            fail = False
            def healthz(self):
                return {"ok":True}
            def detect(self, frame, *, score_threshold=None):
                if self.fail:
                    raise RuntimeError("inference failed")
                return make_result(frame,[.8],[sample_corners()],"fake-test",score_threshold)

        detector = Detector()
        app = create_app(detector)
        app.logger.disabled = True
        server = make_server("127.0.0.1",0,app)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        client = build_cubify_anything({
            "mode":"remote","server_url":f"http://127.0.0.1:{server.server_port}",
            "timeout_s":5})
        client._session.trust_env = False
        try:
            self.assertTrue(client.healthz()["ok"])
            result = client.detect(sample_frame())
            np.testing.assert_array_equal(result.cuboids[0].corners_camera_m, sample_corners())
            self.assertEqual(client.detect(sample_frame(),score_threshold=.9).cuboids,())
            bad = app.test_client().post("/v1/cubify/detect",json={"protocol_version":0})
            self.assertEqual(bad.status_code,422)
            detector.fail = True
            import requests
            with self.assertRaises(requests.HTTPError):
                client.detect(sample_frame())
            client.close()
            with self.assertRaises(RuntimeError):
                client.detect(sample_frame())
        finally:
            client.close()
            server.shutdown()
            thread.join(5)
            server.server_close()
