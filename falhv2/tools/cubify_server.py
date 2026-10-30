"""Run with python falhv2/main.py --task cubify-server --weights ... --source-path ..."""
from __future__ import annotations

import argparse
from urllib.parse import urlparse



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


def main(argv=None, *, model_config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=model_config["path"])
    parser.add_argument("--source-path", default=model_config["source_path"])
    parser.add_argument("--device", default=model_config["device"])
    parser.add_argument("--max-side", type=int, default=model_config["max_side"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=urlparse(model_config["url"]).port or 8093)
    parser.add_argument("--threshold", type=float, default=model_config["score_threshold"])
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1,65535]")
    from ..model_clients import build_cubify_anything
    client = build_cubify_anything({
        "mode": "local", "path": args.weights, "source_path": args.source_path,
        "device": args.device, "max_side": args.max_side, "score_threshold": args.threshold})
    try:
        create_app(client).run(host=args.host, port=args.port, threaded=False,
                               debug=False, use_reloader=False)
    finally:
        client.close()
