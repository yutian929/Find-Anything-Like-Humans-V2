import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from dataclasses import replace

from falhv2 import main as app
from falhv2.tests.test_depth_replacement import frame, grid, prediction, FRONT, BEHIND
from falhv2.map_maintainers.depth_replace import DepthReplacementMap


def wait_for(predicate, timeout=5.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for runtime condition")


class FakeRobot:
    def __init__(self, **options):
        self.counter = 0
        self.closed = False
        self.owner = None

    def __enter__(self):
        self.owner = threading.get_ident()
        return self

    def check_owner(self):
        assert self.owner == threading.get_ident()

    def grid_snapshot(self, **options):
        self.check_owner()
        return grid()

    def capture(self, **options):
        self.check_owner()
        self.counter += 1
        return frame(self.counter, depth=3.)

    def __exit__(self, *exc):
        self.check_owner()
        self.closed = True


class FakeModel:
    def __init__(self):
        self.closed = False
        self.calls = 0

    def detect(self, f):
        self.calls += 1
        return prediction(f, (FRONT, BEHIND))

    def close(self):
        self.closed = True


class LiveSystemTests(unittest.TestCase):
    def runtime(self, folder):
        cfg = SimpleNamespace(**{k:copy.deepcopy(v) for k,v in vars(app.settings).items() if k.isupper()})
        cfg.OBSERVATION_WORKERS = 1
        return app.LiveSystem(cfg, folder, dict(map_id="room", map_revision="session-1"),
                              update_interval_s=.02)

    def scene(self):
        b = DepthReplacementMap()
        b.set_grid(replace(grid(), resolution=.5, rows=tuple(["."*20]*16)))
        f = frame()
        b.integrate(f, prediction(f, (FRONT, BEHIND)))
        return b.to_scene()

    def test_default_entry_dispatches_live_without_offline_planning(self):
        with patch.object(app, "run_live", return_value=0) as live, patch.object(app, "plan") as plan:
            self.assertEqual(app.main([]), 0)
        live.assert_called_once_with([], app.settings)
        plan.assert_not_called()

    def test_fast_keeps_updating_during_slow_and_snapshot_is_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.runtime(folder)
            robot, model = FakeRobot(), FakeModel()
            started, finish = threading.Event(), threading.Event()
            captured = []
            def slow(request_id, scene):
                captured.append(scene)
                started.set()
                if not finish.wait(5):
                    raise TimeoutError("Test slow worker was not released")
                return dict(source_map=dict(revision=scene["mapping"]["revision"]),
                            anchor_count=1, photo_count=1, anchors=[], status="complete")
            with patch("falhv2.robot_adapters.anchor_map.AnchorMapRobot", return_value=robot), patch(
                    "falhv2.model_clients.build_cubify_anything", return_value=model), patch.object(
                    runtime, "_plan_snapshot", side_effect=slow):
                runtime.start()
                try:
                    wait_for(lambda: runtime.state()["fast"]["status"] == "running")
                    self.assertEqual(runtime.state()["slow"]["status"], "idle")
                    self.assertTrue(runtime.request_replan()[0])
                    self.assertTrue(started.wait(2))
                    frozen = copy.deepcopy(captured[0])
                    self.assertFalse(runtime.request_replan()[0])
                    revision = frozen["mapping"]["revision"]
                    wait_for(lambda: runtime.state()["view"]["revision"] >= revision+3)
                    self.assertEqual(captured[0], frozen)
                    finish.set()
                    wait_for(lambda: runtime.state()["slow"]["status"] == "complete")
                    self.assertTrue((Path(folder)/"map.json").exists())
                finally:
                    finish.set()
                    runtime.close()
            self.assertTrue(robot.closed)
            self.assertTrue(model.closed)
            self.assertFalse(any(t.is_alive() for t in runtime.threads))

    def test_slow_failure_does_not_stop_fast_and_request_can_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.runtime(folder)
            robot, model = FakeRobot(), FakeModel()
            with patch("falhv2.robot_adapters.anchor_map.AnchorMapRobot", return_value=robot), patch(
                    "falhv2.model_clients.build_cubify_anything", return_value=model), patch.object(
                    runtime, "_plan_snapshot", side_effect=ValueError("bad camera configuration")):
                runtime.start()
                try:
                    wait_for(lambda: runtime.state()["fast"]["status"] == "running")
                    runtime.request_replan()
                    wait_for(lambda: runtime.state()["slow"]["status"] == "error")
                    revision = runtime.state()["view"]["revision"]
                    wait_for(lambda: runtime.state()["view"]["revision"] > revision)
                    self.assertTrue(runtime.request_replan()[0])
                finally:
                    runtime.close()

    def test_no_valid_map_rejects_replan_and_model_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.runtime(folder)
            self.assertFalse(runtime.request_replan()[0])
            with patch("falhv2.model_clients.build_cubify_anything", side_effect=ValueError("weights missing")):
                runtime.start()
                try:
                    wait_for(lambda: runtime.state()["fast"]["status"] == "error")
                    self.assertIn("weights missing", runtime.state()["fast"]["error"])
                    self.assertFalse(runtime.request_replan()[0])
                finally:
                    runtime.close()

    def test_actual_planner_exports_frozen_source_and_html(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.runtime(folder)
            scene = self.scene()
            with contextlib.redirect_stdout(io.StringIO()):
                result = runtime._plan_snapshot(1, scene)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["source_map"]["revision"], 1)
            output = Path(folder)/"plans/0001"
            self.assertTrue((output/"anchor_plan.html").exists())
            saved = json.loads((output/"anchor_plan.json").read_text())
            self.assertEqual(saved["source_map"], result["source_map"])
            self.assertEqual(json.loads((output/"scene.json").read_text()), json.loads(json.dumps(scene)))

    def test_http_status_command_conflict_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.runtime(folder)
            runtime.scene = self.scene()
            runtime.fast["status"] = "running"
            server = app.make_live_server(runtime, "127.0.0.1", 0, runtime.config.LIVE_PAGE)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url+"/") as response:
                    self.assertIn("重规划锚点", response.read().decode())
                with urlopen(url+"/api/state") as response:
                    self.assertEqual(json.load(response)["slow"]["status"], "idle")
                with urlopen(Request(url+"/api/replan", method="POST")) as response:
                    self.assertEqual(response.status, 202)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url+"/api/replan", method="POST"))
                self.assertEqual(error.exception.code, 409)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url+"/api/replan", method="POST", headers={"Origin":"http://elsewhere"}))
                self.assertEqual(error.exception.code, 403)
                (Path(folder)/"outside.txt").write_text("not a public plan")
                with self.assertRaises(HTTPError) as error:
                    urlopen(url+"/plans/../outside.txt")
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                runtime.close()


if __name__ == "__main__":
    unittest.main()
