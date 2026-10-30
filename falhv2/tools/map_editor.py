"""Serve the local map editor using the Python standard library."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path



class EditorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/map_editor.html"):
            self.send_error(404)
            return
        payload = self.server.page_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main(argv=None, *, page_path=None, map_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page", type=Path, default=page_path, required=page_path is None)
    parser.add_argument("--map-output", type=Path, default=map_path, required=map_path is None)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), EditorHandler)
    server.page_path = Path(args.page)
    print(f"导出地图后请保存到：{args.map_output}", flush=True)
    print(f"打开浏览器：http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
