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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from falhv2 import main as entrypoint
from falhv2.anchor_planners import anchor_verification as verification
from falhv2.anchor_planners.anchor_candidates import yaw_grid
from falhv2.anchor_planners.observation_model import PointCamera
from falhv2.anchor_planners import regions_extractor
from falhv2.config import defaults as settings
from test_anchor_common import config as camera_config
from falhv2.anchor_planners.observation_model import BoxObserver, ObservationSettings
from falhv2.anchor_planners.observation_region import radial_weight
from test_observation_model import box, camera_config as box_camera_config


class ConfigTests(unittest.TestCase):
    def test_config_weights_and_threshold_change_actual_observation(self):
        cfg = box_camera_config()
        target = box('a', [3, -.5, .5], [4, .5, 1.5])
        with patch.dict(settings.OBSERVATION_DEFAULTS,
                        radial_inner_weight=.6, radial_outer_weight=.6, min_observation_score=.55):
            row = BoxObserver(PointCamera(entrypoint.configured_camera(cfg)), [target]).evaluate(0, [[0, 0]], [0])[0]
            self.assertAlmostEqual(row['observation_score'], .6)
            self.assertTrue(row['qualified'])
            with patch.dict(settings.OBSERVATION_DEFAULTS, min_observation_score=.65):
                row = BoxObserver(PointCamera(entrypoint.configured_camera(cfg)), [target]).evaluate(0, [[0, 0]], [0])[0]
                self.assertFalse(row['qualified'])
                self.assertEqual(row['reason'], 'observation_score')
                cfg['observation_quality'].update(radial_inner_weight=.9, radial_outer_weight=.9)
                row = BoxObserver(PointCamera(entrypoint.configured_camera(cfg)), [target]).evaluate(0, [[0, 0]], [0])[0]
                self.assertAlmostEqual(row['observation_score'], .9)
                self.assertTrue(row['qualified'])  # Explicit JSON endpoints override config.py.

    def test_config_radial_weights_preserve_linear_interpolation(self):
        import numpy as np
        with patch.dict(settings.OBSERVATION_DEFAULTS, radial_inner_weight=.8, radial_outer_weight=.4):
            np.testing.assert_allclose(radial_weight(np.array([1, 2, 3]), 1, 3, settings.OBSERVATION_DEFAULTS["radial_inner_weight"], settings.OBSERVATION_DEFAULTS["radial_outer_weight"]), [.8, .6, .4])
            self.assertEqual(radial_weight(2, 2, 2, settings.OBSERVATION_DEFAULTS["radial_inner_weight"], settings.OBSERVATION_DEFAULTS["radial_outer_weight"]), .8)

    def test_config_sampling_and_json_precedence(self):
        with patch.dict(settings.OBSERVATION_DEFAULTS, sample_grid=8, refinement_grid=16,
                        min_in_frame_fraction=.9, min_visible_pixels=500):
            implicit = ObservationSettings.from_config(entrypoint.configured_camera(camera_config()))
            self.assertEqual((implicit.sample_grid, implicit.refinement_grid), (8, 16))
            self.assertEqual((implicit.min_in_frame_fraction, implicit.min_visible_pixels), (.9, 500))
            explicit = ObservationSettings.from_config(entrypoint.configured_camera(box_camera_config()))
            self.assertEqual((explicit.sample_grid, explicit.refinement_grid, explicit.min_visible_pixels), (24, 48, 10))
            observer = BoxObserver(PointCamera(box_camera_config()), [])
            with patch.object(settings, 'AUDIT_SAMPLE_GRID', 16):
                self.assertEqual(verification.dense_observer(observer, [], sample_grid=entrypoint.planning_audit()["sample_grid"]).settings.sample_grid, 48)
            with patch.object(settings, 'AUDIT_SAMPLE_GRID', 64):
                self.assertEqual(verification.dense_observer(observer, [], sample_grid=entrypoint.planning_audit()["sample_grid"]).settings.sample_grid, 64)

    def test_invalid_experiment_parameters_are_rejected(self):
        for key, value in [('radial_outer_weight', 1.1), ('radial_inner_weight', .4),
                           ('min_observation_score', 0), ('sample_grid', True), ('refinement_grid', 8)]:
            with self.subTest(key=key), patch.dict(settings.OBSERVATION_DEFAULTS, {key: value}):
                with self.assertRaises(ValueError):
                    ObservationSettings.from_config(entrypoint.configured_camera(camera_config()))
        for key, value in [('AUDIT_SAMPLE_GRID', 129), ('MAX_AUDIT_RETRIES', -1), ('MAX_AUDIT_RETRIES', True),
                           ('SAME_COST_REPAIR_TIME_LIMIT_S', -1), ('SAME_COST_REPAIR_TIME_LIMIT_S', float('nan'))]:
            with self.subTest(key=key), patch.object(settings, key, value):
                with self.assertRaises(ValueError):
                    entrypoint.planning_audit()
        with patch.object(settings, 'YAW_STEP_DEG', 0):
            with self.assertRaises(ValueError):
                yaw_grid(settings.YAW_STEP_DEG)

    def test_config_planning_defaults_and_cli_overrides_are_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            values = self.prepare(Path(folder))
            with patch.multiple(settings, **values, YAW_STEP_DEG=90,
                                SELECTION_TIME_LIMIT_S=5, AUDIT_SAMPLE_GRID=64, MAX_AUDIT_RETRIES=0):
                for args, step, budget in [([], 90, 5), (['--yaw-step-deg', '45', '--selection-time-limit-s', '10'], 45, 10)]:
                    with patch.object(sys, 'argv', ['main.py', *args]), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(entrypoint.main(), 0)
                    result = entrypoint.read_input(values['ANCHOR_PLAN_PATH'])
                    self.assertEqual(result['configuration']['yaw_step_deg'], step)
                    self.assertEqual(result['statistics']['yaw_count'], 360//step)
                    self.assertEqual(result['configuration']['selection_time_limit_s'], budget)
                    self.assertEqual(result['configuration']['audit'], dict(sample_grid=64, max_retries=0,
                        same_cost_time_limit_s=settings.SAME_COST_REPAIR_TIME_LIMIT_S))

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
                    SAFE_REGIONS_PATH=outputs/'safe_regions.json', SAFE_REGIONS_SVG_PATH=outputs/'safe_regions.svg',
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
                    with patch.object(sys, 'argv', ['regions_extractor.py']), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(entrypoint.main(["--task", "safe-regions"]), 0)
                    (outputs/'map.json').write_text('invalid old output')
                    with patch.object(sys, 'argv', ['main.py']), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(entrypoint.main(), 0)
                    result = entrypoint.read_input(outputs/'anchor_plan.json')
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
                with patch.object(sys, 'argv', ['main.py']), contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as failure:
                        entrypoint.main()
                    self.assertEqual(failure.exception.code, 2)
                    self.assertIn('--config', error.getvalue())
                self.assertFalse(values['OUTPUT_DIR'].exists())
                with patch.object(sys, 'argv', ['main.py', '--scene', str(actual_map)]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(entrypoint.main(), 0)


class ConfigBoundaryTests(unittest.TestCase):
    def test_only_main_imports_project_config(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        violations=[]
        for path in root.rglob("*.py"):
            rel=path.relative_to(root)
            if rel.parts[0] in ("config","tests") or rel.as_posix()=="main.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                names=([node.module or ""] if isinstance(node,ast.ImportFrom)
                       else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
                if any(name=="config" or name.startswith("config.") or name.startswith("falhv2.config")
                       for name in names):
                    violations.append(str(rel))
        self.assertEqual(violations,[])

    def test_modules_import_without_loading_config(self):
        import subprocess
        source = (
            "import sys; import falhv2.model_clients.cubify_anything; "
            "import falhv2.tools.cubify_preview; import falhv2.tools.cubify_server; "
            "import falhv2.anchor_planners.anchor_verification; "
            "import falhv2.tools.map_editor; import falhv2.tools.camera_parameters; "
            "assert not any(n == 'falhv2.config' or n.startswith('falhv2.config.') for n in sys.modules)"
        )
        result=subprocess.run([sys.executable,"-c",source],
                              cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_two_config_files_dispatch_without_global_mutation(self):
        from falhv2.tools import cubify_preview
        before=dict(settings.CUBIFY)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for index in (1,2):
                path=root/f"robot{index}.py"
                path.write_text(f"CUBIFY = {{'mode': 'remote', 'url': 'http://robot{index}:8093'}}\nOUTPUT_DIR = 'results{index}'\n")
                with patch.object(cubify_preview,"main",return_value=0) as run:
                    self.assertEqual(entrypoint.main(["--config",str(path),"--task","cubify-preview",
                                                     "--threshold","0.4"]),0)
                    self.assertEqual(run.call_args.args[0],["--threshold","0.4"])
                    self.assertEqual(run.call_args.kwargs["model_config"]["url"],f"http://robot{index}:8093")
                    self.assertEqual(run.call_args.kwargs["output_dir"],root/f"results{index}")
                    run.call_args.kwargs["model_config"]["mode"]="changed-by-caller"
                self.assertEqual(settings.CUBIFY,before)

    def test_selected_config_controls_actual_planning(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            values=ConfigTests().prepare(root)
            path=root/"experiment.py"
            path.write_text("\n".join(f"{key} = {str(value)!r}" for key,value in values.items())
                            +"\nYAW_STEP_DEG = 90\nAUDIT_SAMPLE_GRID = 64\nMAX_AUDIT_RETRIES = 0\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main(["--config",str(path)]),0)
            result=entrypoint.read_input(values["ANCHOR_PLAN_PATH"])
            self.assertEqual(result["configuration"]["yaw_step_deg"],90)
            self.assertEqual(result["configuration"]["audit"]["sample_grid"],64)
            self.assertEqual(result["configuration"]["audit"]["max_retries"],0)
            self.assertTrue(values["ANCHOR_PLAN_PATH"].with_suffix(".html").exists())


if __name__ == '__main__':
    unittest.main()
