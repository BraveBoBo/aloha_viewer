#!/usr/bin/env python
"""ALOHA / LeRobot dataset viewer — HTTP layer only; all data access lives in dataset.py.

Run with an env that has pandas + PyAV (av) + Pillow (e.g. evo_depth):
    /home/oem/miniconda3/envs/evo_depth/bin/python serve.py
then open http://localhost:8000 and pick a dataset from the catalog.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import dataset

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8",
                           open(os.path.join(HERE, "index.html"), "rb").read())
            elif u.path == "/catalog":
                root = os.path.abspath(q["root"])
                self._json({"root": root, "datasets": [
                    {"path": d, "name": os.path.relpath(d, root)}
                    for d in dataset.dataset_dirs(root)]})
            elif u.path == "/thumb":
                self._send(200, "image/jpeg", dataset.thumb(q["path"]))
            elif u.path == "/load":
                self._json(dataset.meta(q["path"]))
            elif u.path == "/frame":
                self._send(200, "image/jpeg",
                           dataset.frame(q["path"], q["cam"], int(q["idx"])))
            elif u.path == "/series":
                self._json(dataset.series(q["path"], int(q["ep"])))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            if u.path == "/save":
                written, pj = dataset.save_picks(
                    data["dataset_path"], data["out_dir"], data["picks"])
                self._json({"written": written, "picks_json": pj})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"ALOHA dataset viewer → http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
