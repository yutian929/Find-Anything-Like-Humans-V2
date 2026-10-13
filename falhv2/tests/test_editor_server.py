import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_editor import EditorHandler


class EditorServerTests(unittest.TestCase):
    def request(self, path):
        handler = object.__new__(EditorHandler)
        handler.path = path
        handler.wfile = io.BytesIO()
        response = {"headers": {}}
        handler.send_response = lambda status: response.update(status=status)
        handler.send_error = lambda status: response.update(status=status)
        handler.send_header = lambda key, value: response["headers"].update({key: value})
        handler.end_headers = lambda: None
        handler.do_GET()
        return response, handler.wfile.getvalue()

    def test_editor_page(self):
        response, body = self.request("/")
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["headers"]["Cache-Control"], "no-store")
        self.assertIn(b'canvas id="canvas"', body)

    def test_unknown_route(self):
        response, body = self.request("/missing")
        self.assertEqual(response["status"], 404)
        self.assertEqual(body, b"")


if __name__ == "__main__":
    unittest.main()
