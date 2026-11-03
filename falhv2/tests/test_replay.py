import tempfile
import time
import unittest
from unittest.mock import patch
import msgpack
import zmq

from falhv2.robot_adapters.replay import ReplayRobot, ReplayFinished, ReplayStopped
from falhv2.tests.test_robot_adapter import message, map_bytes
from falhv2.tests import test_live_system as live_tests
from falhv2.tests.test_depth_replacement import frame, grid


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.sockets = [self.context.socket(zmq.PUB) for _ in range(2)]
        addresses = [f'tcp://127.0.0.1:{s.bind_to_random_port("tcp://127.0.0.1")}' for s in self.sockets]
        self.robot = ReplayRobot(map_id='test', map_revision='1', endpoint=addresses[0],
                                 map_endpoint=addresses[1], idle_timeout_s=.6)
        self.robot.__enter__()
        time.sleep(.2)

    def tearDown(self):
        self.robot.__exit__()
        for s in self.sockets:
            s.close(0)
        self.context.term()

    def send_map(self, stamp, epoch=1):
        self.sockets[1].send_multipart([b'hermes.map', msgpack.packb(dict(
            request_ns=stamp, response_ns=stamp, epoch=epoch)), map_bytes()])

    def end(self):
        for s in self.sockets:
            s.send_multipart([b'replay.control', msgpack.packb(dict(event='end',published=dict(frame=1,map=1)))])

    def test_historical_time_pairing_no_future_map_and_eof(self):
        self.send_map(10_000_000_000)
        self.send_map(12_000_000_000)
        time.sleep(.05)
        self.sockets[0].send_multipart(message(timestamp_ns=11_000_000_000))
        f,g = self.robot.capture_pair(timeout_s=2)
        self.assertEqual(f.timestamp_ns,11_000_000_000)
        self.assertEqual(g.timestamp_ns,10_000_000_000)
        self.end()
        with self.assertRaises(ReplayFinished):
            self.robot.capture_pair(timeout_s=2)

    def test_future_only_map_is_never_used(self):
        self.send_map(12_000_000_000)
        self.sockets[0].send_multipart(message(timestamp_ns=11_000_000_000))
        self.end()
        with self.assertRaises(ReplayFinished):
            self.robot.capture_pair(timeout_s=2)
        self.assertEqual(self.robot.diagnostics()['unmatched_frames'],1)

    def test_epoch_change_is_fatal(self):
        self.send_map(10_000_000_000)
        self.send_map(11_000_000_000,epoch=2)
        with self.assertRaisesRegex(ReplayStopped,'epoch'):
            self.robot.capture_pair(timeout_s=2)

    def test_missing_end_is_interrupted_not_completed(self):
        self.send_map(10_000_000_000)
        with self.assertRaisesRegex(ReplayStopped,'中断'):
            self.robot.capture_pair(timeout_s=2)

    def test_latest_frame_buffer_is_bounded_and_drained_before_eof(self):
        self.send_map(10_000_000_000)
        time.sleep(.05)
        for i in range(5):
            self.sockets[0].send_multipart(message(frame_id=i,timestamp_ns=10_000_000_001+i))
            time.sleep(.02)
        self.end()
        time.sleep(.1)
        f,g=self.robot.capture_pair(timeout_s=2)
        self.assertEqual(f.frame_id,'4')
        self.assertEqual(self.robot.diagnostics()['overwritten_frames'],4)
        with self.assertRaises(ReplayFinished):
            self.robot.capture_pair(timeout_s=2)


class ReplayRuntimeTests(unittest.TestCase):
    def test_eof_keeps_map_and_allows_replanning_without_live_robot(self):
        class Source:
            def __init__(self, **kw): self.n=0
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def capture_pair(self,**kw):
                self.n+=1
                if self.n>1: raise ReplayFinished()
                return frame(),grid()
            def diagnostics(self): return {'processed_frames':1}
        with tempfile.TemporaryDirectory() as folder:
            runtime=live_tests.LiveSystemTests().runtime(folder)
            runtime.robot_options['source']='replay'
            with patch('falhv2.robot_adapters.replay.ReplayRobot',Source), \
                 patch('falhv2.robot_adapters.anchor_map.AnchorMapRobot') as online, \
                 patch('falhv2.model_clients.build_cubify_anything',return_value=live_tests.FakeModel()):
                # Only fast worker: inspect queued planning request without running MILP.
                runtime._fast_loop()
                self.assertEqual(runtime.state()['fast']['status'],'complete')
                self.assertTrue(runtime.request_replan()[0])
                online.assert_not_called()
