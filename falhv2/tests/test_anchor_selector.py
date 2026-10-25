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
from main import plan
from observation_model import PointCamera
from anchor_candidates import stations_from_regions
from test_anchor_common import config


class AnchorSelectorTests(unittest.TestCase):
    def scene(self):
        return {'name': 'integration', 'grid': {'rows': ['....']*4, 'origin': [0, 0], 'resolution': 1},
                'objects': [{'id': 'target', 'position': [2, 2, 0]}]}

    def robot(self):
        return {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}

    def test_candidates_cover_all_safe_cells_and_actions_retain_camera_coverage(self):
        scene = self.scene()
        result, regions, stations, candidates = plan(scene, self.robot(), config())
        self.assertEqual(result['method'], 'object_driven_lexicographic')
        self.assertEqual(result['status'], 'complete')
        expected = {(r['id'], tuple(cell)) for r in regions['regions'] for cell in r['cells']}
        self.assertEqual({(s.region_id, s.cell) for s in stations}, expected)
        self.assertEqual(len(stations), len(expected))
        self.assertEqual([(s.region_id, s.cell) for s in stations], sorted(expected))
        self.assertEqual(stations, stations_from_regions(regions))
        self.assertEqual(result['schema_version'], 5)
        self.assertEqual(result['navigation'], 'not_planned')
        self.assertEqual(result['start_reachability'], 'not_evaluated_no_start_pose')
        self.assertNotIn('path_length_m', result['statistics'])
        self.assertTrue(all('first_route_index' not in a for a in result['anchors']))
        camera = PointCamera(config())
        for anchor in result['anchors']:
            region = next(r for r in regions['regions'] if r['id'] == anchor['region_id'])
            self.assertIn(anchor['cell'], region['cells'])
            for action in anchor['observations']:
                self.assertTrue(camera.visible(np.array([anchor['position']]), np.array([action['yaw_rad']]), scene['objects'][0]['position'])[0])

    def test_all_regions_are_candidates_without_a_start_pose(self):
        from test_anchor_object_driver import brute_supports
        scene = self.scene()
        scene['grid'] = {'rows': ['..#..']*3, 'origin': [10, -2], 'resolution': 1}
        scene['objects'] = [{'id': 'left', 'position': [10.5, -.5, 0]},
                            {'id': 'right', 'position': [14.5, -.5, 0]}]
        result, regions, stations, candidates = plan(scene, self.robot(), config())
        self.assertEqual(len(regions['regions']), 2)
        self.assertEqual(len(stations), 12)
        self.assertEqual({s.region_id for s in stations}, {1, 2})
        self.assertEqual(stations[0].position, (10.5, .5))
        self.assertEqual(candidates.supports,
                         brute_supports(stations, scene['objects'], PointCamera(config()), candidates.yaws))
        self.assertEqual(result['covered_object_ids'], ['left', 'right'])

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
            run = subprocess.run([sys.executable, '-B', str(ROOT/'main.py'),
                '--scene', str(folder/'scene.json'), '--robot', str(folder/'robot.json'),
                '--camera', str(folder/'camera.json'), '--output', str(output)],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'complete')
            page = output.with_suffix('.html').read_text()
            payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)[1]
            data = json.loads(payload)
            self.assertEqual(len(data['methods']), 2)
            self.assertEqual(data['region_count'], 1)
            self.assertNotIn('path_length_m', data)
            self.assertNotIn('id="path"', page)
            self.assertNotIn('<polyline', output.with_suffix('.svg').read_text())
            self.assertEqual(data['objects'][0]['id'], scene['objects'][0]['id'])
            self.assertNotIn('__ANCHOR_DATA__', page)
            self.assertNotIn('<script>alert(1)</script>', page)
            self.assertTrue(output.with_suffix('.svg').is_file())
            supports = json.loads(output.with_suffix('.supports.json').read_text())
            self.assertTrue(supports['positive_stations'])
            self.assertTrue(supports['supports'])
            self.assertEqual(supports['schema_version'], 4)
            self.assertTrue(all('route_indices' not in s for s in supports['positive_stations']))


if __name__ == '__main__':
    unittest.main()
