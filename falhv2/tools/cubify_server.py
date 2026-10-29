"""Run with python -m falhv2.tools.cubify_server --weights ... --source-path ..."""
from __future__ import annotations

import argparse


def create_app(client):
    from flask import Flask, jsonify, request
    from ..model_clients.cubify_anything import decode_frame, encode_result
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

    @app.get("/healthz")
    def health():
        status = client.healthz()
        return jsonify(status), 200 if status["ok"] else 503

    @app.post("/v1/cubify/detect")
    def detect():
        try:
            frame, score = decode_frame(request.get_json())
            result = client.detect(frame, score_threshold=score)
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 422
        except RuntimeError as exc:
            app.logger.exception("CuTR inference failed")
            return jsonify(error=str(exc)), 503
        return jsonify(encode_result(result))

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args(argv)
    from ..model_clients import build_cubify_anything
    client = build_cubify_anything({
        "mode": "local", "path": args.weights, "source_path": args.source_path,
        "device": args.device, "max_side": args.max_side})
    try:
        create_app(client).run(host=args.host, port=args.port, threaded=False,
                               debug=False, use_reloader=False)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
