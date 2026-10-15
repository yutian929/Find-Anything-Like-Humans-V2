"""Integration checks for the production object-driven entry point."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anchor_selector import plan, PointCamera
from test_anchor_common import config


class AnchorSelectorTests(unittest.TestCase):
    def scene(self):
        return {'name': 'integration', 'grid': {'rows': ['....']*4, 'origin': [0, 0], 'resolution': 1},
                'objects': [{'id': 'target', 'position': [2, 2, 0]}]}

    def robot(self):
        return {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}

    def test_actions_retain_path_membership_and_camera_coverage(self):
        scene = self.scene()
        result, path, stations, candidates = plan(scene, self.robot(), config())
        self.assertEqual(result['method'], 'object_driven')
        self.assertEqual(result['status'], 'complete')
        camera = PointCamera(config())
        for anchor in result['anchors']:
            region = next(r for r in path['regions'] if r['id'] == anchor['region_id'])
            self.assertIn(anchor['cell'], region['route_cells'])
            for action in anchor['observations']:
                self.assertTrue(camera.visible(np.array([anchor['position']]), np.array([action['yaw_rad']]), scene['objects'][0]['position'])[0])

    def test_partial_and_empty_keep_the_input_target_denominator(self):
        scene = self.scene()
        scene['objects'].append({'id': 'far', 'position': [100, 100, 0]})
        result, *_ = plan(scene, self.robot(), config())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['uncovered_object_ids'], ['far'])
        self.assertEqual(result['coverage_fraction'], .5)
        scene['objects'] = []
        result, *_ = plan(scene, self.robot(), config())
        self.assertEqual(result['status'], 'empty_targets')
        self.assertEqual(result['anchor_count'], 0)
        self.assertIsNone(result['coverage_fraction'])

    def test_invalid_inputs_are_rejected(self):
        camera = copy.deepcopy(config())
        camera['mount']['pitch_deg'] = None
        with self.assertRaises(ValueError):
            plan(self.scene(), self.robot(), camera)
        with self.assertRaises(ValueError):
            plan(self.scene(), self.robot(), config(), yaw_step_deg=0)

    def test_cli_exports_offline_html_and_sparse_supports(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            scene = self.scene()
            scene['objects'][0]['id'] = '</script><script>alert(1)</script>'
            for name, data in [('scene', scene), ('robot', self.robot()), ('camera', config())]:
                (folder/f'{name}.json').write_text(json.dumps(data))
            output = folder/'anchor_plan.json'
            run = subprocess.run([sys.executable, '-B', str(ROOT/'anchor_selector.py'),
                '--scene', str(folder/'scene.json'), '--robot', str(folder/'robot.json'),
                '--camera', str(folder/'camera.json'), '--output', str(output), '--save-supports'],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'complete')
            page = output.with_suffix('.html').read_text()
            payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)[1]
            data = json.loads(payload)
            self.assertEqual(len(data['methods']), 1)
            self.assertEqual(data['objects'][0]['id'], scene['objects'][0]['id'])
            self.assertNotIn('__ANCHOR_DATA__', page)
            self.assertNotIn('<script>alert(1)</script>', page)
            self.assertTrue(output.with_suffix('.svg').is_file())
            supports = json.loads(output.with_suffix('.supports.json').read_text())
            self.assertTrue(supports['positive_stations'])
            self.assertTrue(supports['supports'])


if __name__ == '__main__':
    unittest.main()
