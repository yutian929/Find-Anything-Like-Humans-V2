"""The unified entry point exports every stage through script and package entry points."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from falhv2 import main as entrypoint
from falhv2.config import defaults as settings
from falhv2.anchor_planners import regions_extractor
from falhv2.anchor_planners import anchor_candidates
from falhv2.anchor_planners.observation_model import BoxObserver
from test_observation_model import box, camera_config
from test_anchor_common import config


class MainTests(unittest.TestCase):
    def inputs(self, folder):
        scene = {'name': 'entry point integration',
                 'grid': {'rows': ['....'] * 4, 'origin': [0, 0], 'resolution': 1},
                 'objects': [{'id': 'target', 'position': [2, 2, 0]}]}
        robot = {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2,
                           'safety_margin_m': 0}}
        paths = {}
        for name, data in [('scene', scene), ('robot', robot), ('camera', config())]:
            paths[name] = folder / f'{name}.json'
            paths[name].write_text(json.dumps(data), encoding='utf-8')
        return scene, robot, paths

    def test_offline_task_exports_all_stages_from_one_region_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            scene, robot, inputs = self.inputs(folder)
            before = {p: p.read_bytes() for p in inputs.values()}
            output = folder / 'outputs' / 'anchor_plan.json'
            regions = folder / 'a1' / 'regions.json'
            svg = folder / 'diagnostics' / 'regions.svg'
            with patch.multiple(settings, MAP_PATH=inputs['scene'], ROBOT_PATH=inputs['robot'],
                                CAMERA_PATH=inputs['camera'], ANCHOR_PLAN_PATH=output,
                                SAFE_REGIONS_PATH=regions, SAFE_REGIONS_SVG_PATH=svg), \
                    patch.object(entrypoint, 'find_safe_regions', wraps=regions_extractor.plan) as extract, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main(['--task', 'anchors']), 0)
            extract.assert_called_once_with(scene, robot)
            for path in [output, output.with_suffix('.html'), output.with_suffix('.svg'),
                         output.with_suffix('.supports.json'), regions, svg]:
                self.assertTrue(path.is_file(), path)
            self.assertEqual(json.loads(regions.read_text()), regions_extractor.plan(scene, robot))
            result = json.loads(output.read_text())
            self.assertEqual(result['selected_cost'], [1, 1])
            self.assertEqual(result['covered_object_ids'], ['target'])
            self.assertEqual(result['optimization']['geometry_verification'], 'passed')
            self.assertEqual(before, {p: p.read_bytes() for p in inputs.values()})

    def test_script_and_package_cli_work_with_custom_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _, _, inputs = self.inputs(folder)
            variants = [([str(ROOT / 'main.py')], folder),
                        (['-m', 'falhv2.main'], ROOT.parent)]
            for index, (command, cwd) in enumerate(variants):
                with self.subTest(command=command):
                    # An A1-looking plan name must not be overwritten by the A1 export.
                    output = folder / str(index) / 'safe_regions.json'
                    run = subprocess.run([sys.executable, '-B', *command, '--task', 'anchors',
                        '--scene', str(inputs['scene']), '--robot', str(inputs['robot']),
                        '--camera', str(inputs['camera']), '--output', str(output),
                        '--no-save-supports'], cwd=cwd, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    result = json.loads(output.read_text())
                    self.assertEqual(result['schema_version'], 5)
                    self.assertEqual(result['selected_cost'], [1, 1])
                    self.assertTrue(output.with_suffix('.safe_regions.json').is_file())
                    self.assertTrue(output.with_suffix('.safe_regions.svg').is_file())
                    self.assertFalse(output.with_suffix('.supports.json').exists())

    def test_native_candidate_option_keeps_float64_final_verifier(self):
        scene = {'grid': {'rows': ['....']*4, 'origin': [0, 0], 'resolution': 1},
                 'objects': [box('target', [2.5, 1.5, .5], [3.5, 2.5, 1.5])]}
        robot = {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}
        # Exercise dispatch without requiring the optional package in this test.
        def candidates(*args, **kwargs):
            self.assertEqual(kwargs.pop('ray_backend'), 'open3d')
            return anchor_candidates.generate_candidates(*args, **kwargs)
        with patch.object(entrypoint, 'generate_candidates', side_effect=candidates), \
                patch.object(entrypoint, 'BoxObserver', wraps=BoxObserver) as verifier:
            result, *_ = entrypoint.plan(scene, robot, camera_config(), ray_backend='open3d')
        self.assertEqual(verifier.call_args.kwargs['ray_backend'], 'numpy')
        self.assertEqual(result['configuration']['candidate_ray_backend'], 'open3d')
        self.assertEqual(result['configuration']['verification_ray_backend'], 'numpy')
        self.assertEqual(result['optimization']['geometry_verification'], 'passed')


if __name__ == '__main__':
    unittest.main()
