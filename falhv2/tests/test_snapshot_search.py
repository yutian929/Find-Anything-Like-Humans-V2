import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from falhv2.model_clients.valen import Valen
from falhv2.robot_adapters.base import CameraIntrinsics, RGBDFrame
from falhv2.object_search.search import find_object
from falhv2.snapshots_updater.store import initialize, read_json, record, search_views, view_path, write_json


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        scene, plan = self.root / 'scene.json', self.root / 'plan.json'
        write_json(scene, {'name': 'map', 'objects': []})
        write_json(plan, {'scene_name': 'map', 'covered_object_ids': [], 'uncovered_object_ids': [],
                          'anchors': [dict(station_index=23, position=[0, 0], observations=[
                              dict(yaw_index=i, yaw_rad=0, object_ids=[]) for i in (2, 5)])]})
        self.store = self.root / 'store'
        idx = initialize(self.store, scene, plan)
        self.frame = RGBDFrame('f-1', 1, 'test', idx['map_id'], idx['map_revision'], 'cam',
                               CameraIntrinsics(16, 16, 10, 10, 8, 8), np.full((16,16,3), 100, np.uint8),
                               np.ones((16,16), np.float32), np.eye(4), np.eye(4))
        self.view = self.store / view_path(23, 2)

    def capture(self, n, **updates):
        return record(self.store, 23, 2, f'visit-{n}', replace(self.frame, frame_id=f'f-{n}',
                      timestamp_ns=n, **updates))

    def test_visits_pin_replacement_and_roll_back_failed_write(self):
        self.assertEqual(self.capture(1), 'first')
        self.assertEqual(record(self.store, 23, 2, 'visit-1', replace(self.frame, timestamp_ns=2)), 'same_visit')
        self.assertFalse((self.view / 'last').exists())
        self.assertEqual(self.capture(3), 'last')
        with search_views(self.store) as (_, latest, history):
            self.assertEqual(self.capture(4), 'last')
            self.assertEqual(read_json(latest[0][2] / 'meta.json')['timestamp_ns'], 3)
            self.assertEqual(read_json(history[0][2] / 'meta.json')['timestamp_ns'], 1)
        self.assertEqual(self.capture(2), 'stale_frame')
        with patch('falhv2.snapshots_updater.store.cv2.imwrite', return_value=False), self.assertRaises(OSError):
            self.capture(5)
        self.assertEqual(read_json(self.view / 'last/meta.json')['frame_id'], 'f-4')
        self.assertEqual(len(list(self.view.glob('.capture-*'))), 1)

    def test_pose_revision_and_clock_validation(self):
        bad = np.eye(4)
        bad[0,3] = .15
        for change in (dict(T_map_base=bad), dict(map_revision='other')):
            with self.assertRaises(ValueError):
                self.capture(1, **change)
        self.capture(1)
        with self.assertRaises(ValueError):
            self.capture(2, clock_domain='new')
        self.assertFalse((self.view / 'last').exists())

    def test_historical_fallback_and_missing_depth(self):
        depth = self.frame.depth_m.copy()
        depth[3:13,3:13] = 0
        self.capture(1, depth_m=depth)
        self.capture(2, rgb=np.zeros_like(self.frame.rgb))
        client = Valen('unused', 'contract-test')
        def answers(image, questions):
            if questions[0]['name'] == 'identity':
                return {'identity': {'choice': 'match'}}
            return {'present': {'probability': float(image.mean() > 0)},
                    'visibility': {'score': 2}, 'region': {'bbox': [.1,.1,.9,.9], 'confidence': .9}}
        with patch.object(client, 'decisions', side_effect=answers):
            out = find_object(self.store, 'item', client, self.root / 'out')
        self.assertEqual(out['status'], 'found_historical')
        self.assertEqual(out['pending_views'], [[23,5]])
        self.assertEqual(out['match']['frame_id'], 'f-1')
        self.assertIsNone(out['match']['map_xyz_m_estimate'])
        self.assertTrue((self.root / 'out/bbox.png').exists())

    def test_empty_store_and_service_failure(self):
        client = Valen('unused', 'contract-test')
        out = find_object(self.store, 'target', client, self.root / 'empty')
        self.assertEqual(out['status'], 'no_snapshots')
        self.capture(1)
        with patch.object(client, 'inspect', side_effect=RuntimeError('service down')), self.assertRaises(RuntimeError):
            find_object(self.store, 'item', client, self.root / 'failure')
        self.assertFalse((self.root / 'failure').exists())
