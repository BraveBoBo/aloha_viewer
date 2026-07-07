#!/usr/bin/env python
"""ALOHA / LeRobot dataset viewer — 3-camera frames + action curves + keyframe picking.

Run with an env that has pandas + PyAV (av) + Pillow (e.g. evo_depth):
    /home/oem/miniconda3/envs/evo_depth/bin/python serve.py
then open http://localhost:8000 and type a dataset dir, e.g. /data/datasets/piper/block

Videos are AV1; browsers can't seek them frame-accurately, and per-frame random decode via
imageio index= is pathologically slow. But a *sequential* PyAV decode is ~1 ms/frame, so on
load we decode each camera's mp4(s) once into in-memory JPEGs and serve any global frame
index instantly. Episode frame ranges come from the data parquet's global `index` column
(== frame position in the camera's mp4s laid end-to-end); per-episode timestamp/frame_index
reset and must NOT be used for seeking.

Both LeRobot layouts are supported: v3.0 (one concatenated file-000.mp4 per camera) and
v2.1 (one mp4 per episode; concatenating them in `index`-start order tiles the global
index space exactly — verified per-episode mp4 frame count == parquet row count).
"""
import glob
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import av
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
_datasets = {}
_thumbs = {}
_lock = threading.Lock()


def _dataset_dirs(root):
    """Dirs under root that hold a meta/info.json. Doesn't descend into a found dataset."""
    root = os.path.abspath(root)
    out = []
    for dp, dns, _ in os.walk(root):
        if os.path.isfile(os.path.join(dp, "meta", "info.json")):
            out.append(dp)
            dns[:] = []                     # a dataset is a leaf; don't walk its data/videos
        else:
            dns[:] = [d for d in dns if not d.startswith(".")]   # skip hidden dirs
    return sorted(out)


def _thumb(path):
    """First frame of the first camera as a small JPEG (cached). Decodes one frame, cheap."""
    path = os.path.abspath(path)
    if path in _thumbs:
        return _thumbs[path]
    info = json.load(open(os.path.join(path, "meta", "info.json")))
    tmpl = info.get("video_path",
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    cam_keys = [k for k in info.get("features", {}) if k.startswith("observation.images")]
    if not cam_keys:
        raise ValueError("no observation.images features")
    # str.format ignores unused kwargs -> one call covers both v3.0 and v2.1 templates
    mp4 = os.path.join(path, tmpl.format(video_key=cam_keys[0], chunk_index=0, file_index=0,
                                         episode_chunk=0, episode_index=0))
    c = av.open(mp4)
    fr = next(c.decode(c.streams.video[0]))
    c.close()
    buf = io.BytesIO()
    Image.fromarray(fr.to_ndarray(format="rgb24")).save(buf, "JPEG", quality=80)
    _thumbs[path] = buf.getvalue()
    return _thumbs[path]


def _decode_all(mp4s, encode="jpeg"):
    """Sequential decode of mp4(s) -> flat list indexed by global frame position. ~1 ms/frame."""
    out = []
    for mp4 in mp4s:
        c = av.open(mp4)
        s = c.streams.video[0]
        for fr in c.decode(s):
            img = fr.to_ndarray(format="rgb24")
            if encode == "jpeg":
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, "JPEG", quality=88)
                out.append(buf.getvalue())
            else:
                out.append(img)
        c.close()
    return out


def video_paths(ds, cam):
    """mp4 path(s) for a camera in global-frame order: one concatenated file (v3.0 layout),
    or one per episode sorted by global start index (v2.1 layout)."""
    tmpl, key = ds["tmpl"], ds["cam_keys"][cam]
    if "{episode" in tmpl:                       # v2.1: videos/chunk-XXX/<cam>/episode_XXXXXX.mp4
        return [os.path.join(ds["path"], tmpl.format(
                    video_key=key, episode_index=e["i"],
                    episode_chunk=e["i"] // ds["chunk_size"]))
                for e in sorted(ds["episodes"], key=lambda x: x["start"])]
    return [os.path.join(ds["path"], tmpl.format(video_key=key, chunk_index=0, file_index=0))]


def load_dataset(path):
    path = os.path.abspath(path)
    with _lock:
        cached = _datasets.get(path)
        if cached is not None and "frames" in cached:   # frames evicted -> full reload
            return cached

    info_p = os.path.join(path, "meta", "info.json")
    if not os.path.isfile(info_p):
        raise FileNotFoundError(f"no meta/info.json under {path}")
    info = json.load(open(info_p))
    fps = info.get("fps", 30)
    tmpl = info.get("video_path",
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    cam_keys = [k for k in info.get("features", {}) if k.startswith("observation.images")]
    cams = [k.split(".")[-1] for k in cam_keys]

    def dim_names(key, n):
        nm = info.get("features", {}).get(key, {}).get("names")
        if nm and isinstance(nm[0], list):      # LeRobot nests as [[...]]
            nm = nm[0]
        return list(nm) if nm and len(nm) == n else [f"a{i}" for i in range(n)]

    dfiles = sorted(glob.glob(os.path.join(path, "data", "**", "*.parquet"), recursive=True))
    if not dfiles:
        raise FileNotFoundError(f"no data parquet under {path}")
    df = pd.concat([pd.read_parquet(f) for f in dfiles], ignore_index=True)

    lang = {}
    tp = os.path.join(path, "meta", "tasks.parquet")
    tj = os.path.join(path, "meta", "tasks.jsonl")
    if os.path.isfile(tp):              # v3.0
        t = pd.read_parquet(tp)
        if "task_index" in t.columns:   # task string is the index, task_index a column
            lang = {int(v): str(k) for k, v in zip(t.index, t["task_index"])}
        else:
            lang = {i: str(k) for i, k in enumerate(t.index)}
    elif os.path.isfile(tj):            # v2.1: {"task_index": 0, "task": "..."} per line
        for line in open(tj):
            if line.strip():
                r = json.loads(line)
                lang[int(r["task_index"])] = str(r["task"])

    eps = []
    for e, g in df.groupby("episode_index"):
        ti = int(g["task_index"].iloc[0]) if "task_index" in g else 0
        eps.append({"i": int(e), "start": int(g["index"].min()), "end": int(g["index"].max()),
                    "n": int(len(g)), "language": lang.get(ti, "")})

    ds = {
        "path": path, "name": os.path.basename(path), "fps": fps, "cameras": cams,
        "cam_keys": dict(zip(cams, cam_keys)), "tmpl": tmpl, "episodes": eps, "df": df,
        "chunk_size": info.get("chunks_size", 1000),
        "action_dim": len(df["action"].iloc[0]),
        "state_dim": len(df["observation.state"].iloc[0]) if "observation.state" in df else 0,
    }
    ds["action_names"] = dim_names("action", ds["action_dim"])
    ds["state_names"] = dim_names("observation.state", ds["state_dim"])
    # decode every camera once (in-memory JPEGs). Bound memory: keep frames for this dataset
    # only, drop other datasets' frames.
    ds["frames"] = {cam: _decode_all(video_paths(ds, cam)) for cam in cams}
    with _lock:
        for other in _datasets.values():
            other.pop("frames", None)
        _datasets[path] = ds
    return ds


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
                    {"path": d, "name": os.path.relpath(d, root)} for d in _dataset_dirs(root)]})
            elif u.path == "/thumb":
                self._send(200, "image/jpeg", _thumb(q["path"]))
            elif u.path == "/load":
                ds = load_dataset(q["path"])
                self._json({k: ds[k] for k in
                            ("name", "fps", "cameras", "episodes", "action_dim", "state_dim",
                             "action_names", "state_names", "path")})
            elif u.path == "/frame":
                ds = load_dataset(q["path"])            # guarantees frames present
                self._send(200, "image/jpeg", ds["frames"][q["cam"]][int(q["idx"])])
            elif u.path == "/series":
                ds = load_dataset(q["path"])
                g = ds["df"][ds["df"]["episode_index"] == int(q["ep"])]
                out = {"start": int(g["index"].min()),
                       "action": np.stack(g["action"].values).astype(float).round(5).tolist()}
                if "observation.state" in g:
                    out["state"] = np.stack(g["observation.state"].values).astype(float).round(5).tolist()
                self._json(out)
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
                ds = load_dataset(data["dataset_path"])
                out_dir = data["out_dir"]
                os.makedirs(out_dir, exist_ok=True)
                picks = sorted(int(x) for x in data["picks"])
                written = []
                # one lossless PNG per (pick, camera); re-decode from mp4 (sequential, cheap)
                for cam in ds["cameras"]:
                    want = set(picks)
                    gi = 0                      # global frame position across all mp4s
                    for mp4 in video_paths(ds, cam):
                        if not want:
                            break
                        c = av.open(mp4)
                        s = c.streams.video[0]
                        for fr in c.decode(s):
                            if gi in want:
                                fp = os.path.join(out_dir, f"f{gi:06d}_{cam}.png")
                                Image.fromarray(fr.to_ndarray(format="rgb24")).save(fp)
                                written.append(fp)
                                want.discard(gi)
                            gi += 1
                            if not want:
                                break
                        c.close()
                rows = {}
                for idx in picks:
                    r = ds["df"][ds["df"]["index"] == idx]
                    if len(r):
                        rr = r.iloc[0]
                        rows[str(idx)] = {"episode": int(rr["episode_index"]),
                                          "frame_index": int(rr["frame_index"]),
                                          "action": [round(float(x), 5) for x in rr["action"]]}
                pj = os.path.join(out_dir, "picks.json")
                json.dump({"dataset": ds["path"], "picks": rows}, open(pj, "w"), indent=2)
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
