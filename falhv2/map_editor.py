"""Serve the local map editor using the Python standard library."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
try:
    from . import config as settings
except ImportError:
    import config as settings

PAGE = settings.EDITOR_PAGE


class EditorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/map_editor.html"):
            self.send_error(404)
            return
        payload = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), EditorHandler)
    print(f"导出地图后请保存到：{settings.MAP_PATH}", flush=True)
    print(f"打开浏览器：http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
