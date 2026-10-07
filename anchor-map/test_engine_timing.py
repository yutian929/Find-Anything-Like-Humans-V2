"""检索与检测响应的端到端引擎计时。"""

import time
import unittest

from engine import Engine, _Request


class _ImmediateQueue:
    def put(self, request):
        time.sleep(0.02)
        request.future.set_result({"success": True, "found": False})


class EngineTimingTest(unittest.TestCase):
    def test_query_includes_elapsed_time(self):
        engine = Engine.__new__(Engine)
        engine._queue = _ImmediateQueue()
        result = engine._submit(_Request("query", {"object": "椅子"}))
        self.assertGreaterEqual(result["elapsed_s"], 0.02)

    def test_control_response_has_no_retrieval_time(self):
        engine = Engine.__new__(Engine)
        engine._queue = _ImmediateQueue()
        result = engine._submit(_Request("build"))
        self.assertNotIn("elapsed_s", result)


if __name__ == "__main__":
    unittest.main()
