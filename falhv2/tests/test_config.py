"""Configured paths control inputs; output deletion never changes their selection."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anchor_selector
import path_planner
import config as settings
from test_anchor_common import config as camera_config


class ConfigTests(unittest.TestCase):
    def prepare(self, root):
        data = root/'data'
        data.mkdir()
        scene = {'name': 'persistent input',
                 'grid': {'rows': ['....']*4, 'origin': [0, 0], 'resolution': 1},
                 'objects': [{'id': 'target', 'position': [2, 2, 0]}]}
        robot = {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}
        # Deliberately use custom filenames: entry points must honor config.py.
        for name, value in [('scene.custom.json', scene), ('robot.custom.json', robot), ('camera.custom.json', camera_config())]:
            (data/name).write_text(json.dumps(value))
        outputs = root/'outputs'
        return dict(MAP_PATH=data/'scene.custom.json', ROBOT_PATH=data/'robot.custom.json',
                    CAMERA_PATH=data/'camera.custom.json', OUTPUT_DIR=outputs,
                    PATH_PLAN_PATH=outputs/'path_plan.json', PATH_SVG_PATH=outputs/'path_diagnostic.svg',
                    ANCHOR_PLAN_PATH=outputs/'anchor_plan.json')

    def test_configured_inputs_regenerate_outputs_after_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = self.prepare(root)
            outputs = values['OUTPUT_DIR']
            original = {p: p.read_bytes() for p in (root/'data').iterdir()}
            with patch.multiple(settings, **values):
                for _ in range(2):
                    self.assertFalse(outputs.exists())
                    with patch.object(sys, 'argv', ['path_planner.py']), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(path_planner.main(), 0)
                    (outputs/'map.json').write_text('invalid old output')
                    with patch.object(sys, 'argv', ['anchor_selector.py']), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(anchor_selector.main(), 0)
                    result = settings.read_input(outputs/'anchor_plan.json')
                    self.assertEqual(result['scene_name'], 'persistent input')
                    self.assertEqual(result['covered_object_ids'], ['target'])
                    self.assertTrue((outputs/'anchor_plan.html').is_file())
                    self.assertEqual(original, {p: p.read_bytes() for p in (root/'data').iterdir()})
                    shutil.rmtree(outputs)

    def test_missing_configured_input_fails_and_explicit_override_works(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = self.prepare(root)
            actual_map = values['MAP_PATH']
            values['MAP_PATH'] = root/'missing.json'
            with patch.multiple(settings, **values):
                with patch.object(sys, 'argv', ['anchor_selector.py']), contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as failure:
                        anchor_selector.main()
                    self.assertEqual(failure.exception.code, 2)
                    self.assertIn('config.py', error.getvalue())
                self.assertFalse(values['OUTPUT_DIR'].exists())
                with patch.object(sys, 'argv', ['anchor_selector.py', '--scene', str(actual_map)]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(anchor_selector.main(), 0)


if __name__ == '__main__':
    unittest.main()
