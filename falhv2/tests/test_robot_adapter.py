import json
import struct
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgpack
import numpy as np
import zmq

from falhv2.robot_adapters.anchor_map import (
    AnchorMapRobot, decode_explore_map, decode_rgbd_pose_message)


def message(**changes):
    meta = dict(version=2,type="rgbd_pose",frame_id=17,timestamp_ns=time.time_ns(),
                pose_valid=True,sync={"valid":True},width=8,height=4,
                depth_width=8,depth_height=4,rgb_encoding="rgb8",depth_encoding="z16",
                rgb_dtype="uint8",depth_dtype="uint16",depth_aligned_to="color",
                T_map_camera_frame="color_optical",depth_frame="color_optical",
                depth_scale_m=.001,color_intr=dict(fx=10,fy=11,ppx=4,ppy=2),
                T_map_camera=np.eye(4).tolist(),T_map_base=np.eye(4).tolist())
    meta.update(changes)
    rgb = np.full((4,8,3),[1,2,200],np.uint8)
    depth = np.full((4,8),1500,dtype="<u2")
    return [b"rgbd.pose",msgpack.packb(meta,use_bin_type=True),rgb.tobytes(),depth.tobytes()]


def map_bytes(wrapper=False):
    cells = bytes([0,127,1,1,0,127])
    return (struct.pack("<ffiif",-2.,3.,3,2,.05)
            +(bytes(12)+struct.pack("<I",6) if wrapper else b"")+cells)


class RobotDecoderTests(unittest.TestCase):
    def test_rgbd_units_color_timestamp_and_synchronized_base(self):
        stamp = 1793238647890102272
        T = np.eye(4); T[0,3] = 3
        f = decode_rgbd_pose_message(message(timestamp_ns=stamp,T_map_base=T.tolist()),
                                     map_id="test",map_revision="1")
        self.assertEqual(f.timestamp_ns,stamp)
        np.testing.assert_array_equal(f.rgb[0,0],[1,2,200])
        np.testing.assert_allclose(f.depth_m,1.5)
        np.testing.assert_array_equal(f.T_map_base,T)
        self.assertEqual(f.T_map_camera[0,3],0)

    def test_invalid_protocol_pose_calibration_and_payload(self):
        for change in (dict(pose_valid=False),dict(sync={"valid":False}),
                       dict(sync=[1]),dict(version=1),dict(depth_aligned_to="depth"),
                       dict(T_map_base=None),dict(T_map_camera=np.zeros((4,4)).tolist()),
                       dict(depth_scale_m=0),dict(map_revision="different"),
                       dict(color_intr=dict(fx=10,fy=11,ppx=4,ppy=2,coeffs=[.1])),
                       dict(timestamp_ns=0)):
            with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):
                decode_rgbd_pose_message(message(**change),map_id="test",map_revision="1")
        parts=message();parts[-1]=b"bad"
        with self.assertRaises(ValueError):
            decode_rgbd_pose_message(parts,map_id="test",map_revision="1")

    def test_grid_orientation_origin_unknown_and_versions(self):
        for wrapper in (False,True):
            raw=map_bytes(wrapper)
            grid=decode_explore_map(raw,map_id="test",map_revision="1",timestamp_ns=10)
            self.assertEqual(grid.rows,(".?.","?.."))
            self.assertEqual(grid.origin,(-2.,3.))
            self.assertAlmostEqual(grid.resolution,.05)
            changed=decode_explore_map(raw[:-1]+b"\x00",map_id="test",map_revision="1",timestamp_ns=20)
            self.assertNotEqual(grid.content_version,changed.content_version)
            self.assertEqual(grid.map_revision,changed.map_revision)
        for raw in (b"short",map_bytes()[:-1],map_bytes()+b"extra",
                    struct.pack("<ffiif",0,0,-1,2,.05)):
            with self.assertRaises(ValueError):
                decode_explore_map(raw,map_id="test",map_revision="1",timestamp_ns=1)

    def test_slamtec_signed_evidence_and_unknown_exterior(self):
        # Bottom row: unknown, weak free, strong free; top: signed negatives.
        raw = struct.pack("<ffiif",0.,0.,3,2,.05)+bytes([0,1,127,128,129,255])
        grid = decode_explore_map(raw,map_id="room",map_revision="1",timestamp_ns=1)
        self.assertEqual(grid.rows, ("###", "?.."))
        self.assertEqual(grid.rows[-1][2], ".")  # robot raw 127 is free
        self.assertEqual(grid.rows[-1][0], "?")  # unobserved exterior is NOT traversable

    def test_lifecycle_and_configuration(self):
        robot=AnchorMapRobot(map_id="test",map_revision="1")
        self.assertIsNone(robot._socket)
        with self.assertRaises(RuntimeError):
            robot.capture()
        robot.close()
        with self.assertRaises(ValueError):
            AnchorMapRobot(map_id="",map_revision="1")


class RobotTransportTests(unittest.TestCase):
    def test_real_zmq_fresh_frame_and_stale_timeout(self):
        stop,ready=threading.Event(),threading.Event()
        stale=threading.Event()
        endpoint=[]
        def publisher():
            context=zmq.Context();socket=context.socket(zmq.PUB)
            port=socket.bind_to_random_port("tcp://127.0.0.1")
            endpoint.append(f"tcp://127.0.0.1:{port}");ready.set()
            try:
                while not stop.wait(.01):
                    socket.send_multipart(message(timestamp_ns=time.time_ns()-(10_000_000_000 if stale.is_set() else 0)))
            finally:
                socket.close(0);context.term()
        worker=threading.Thread(target=publisher,daemon=True);worker.start()
        self.assertTrue(ready.wait(2))
        robot=AnchorMapRobot(map_id="test",map_revision="1",endpoint=endpoint[0],max_frame_age_s=.1)
        try:
            robot.open()
            self.assertEqual(robot.capture(timeout_s=2).frame_id,"17")
            stale.set()
            # Let all already-published fresh frames exceed the age limit.
            time.sleep(.15)
            with self.assertRaises(TimeoutError):
                robot.capture(timeout_s=.2)
            robot.close()
            with self.assertRaises(RuntimeError):
                robot.capture()
        finally:
            robot.close();stop.set();worker.join(2)

    def test_real_http_map_refresh_pose_and_failures(self):
        class Handler(BaseHTTPRequestHandler):
            count=0
            fail=False
            def log_message(self,*args):
                pass
            def do_GET(self):
                if self.fail:
                    self.send_error(503);return
                if self.path.endswith("/maps/explore"):
                    type(self).count+=1
                    payload=map_bytes(True)
                    if self.count>1:
                        payload=payload[:-1]+b"\x00"
                elif self.path.endswith("/localization/pose"):
                    payload=json.dumps(dict(x=1,y=-2,yaw=.4,roll=.2,pitch=.3)).encode()
                else:
                    self.send_error(404);return
                self.send_response(200);self.send_header("Content-Length",str(len(payload)))
                self.end_headers();self.wfile.write(payload)
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        robot=AnchorMapRobot(map_id="test",map_revision="1",
                             chassis_url=f"http://127.0.0.1:{server.server_port}")
        try:
            with robot:
                first=robot.grid_snapshot();second=robot.grid_snapshot()
                self.assertNotEqual(first.content_version,second.content_version)
                pose=robot.localization()
                self.assertEqual((pose.x,pose.y),(1.,-2.))
                self.assertAlmostEqual(pose.yaw,.4)
                np.testing.assert_allclose(pose.T_map_base[:3,:3].T@pose.T_map_base[:3,:3],np.eye(3),atol=1e-12)
                self.assertEqual(pose.timestamp_source,"http_midpoint")
                self.assertGreater(pose.rtt_s,0)
                Handler.fail=True
                import requests
                with self.assertRaises(requests.HTTPError):
                    robot.localization()
        finally:
            robot.close();server.shutdown();worker.join(2);server.server_close()
