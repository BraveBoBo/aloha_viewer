# ALOHA / LeRobot Dataset Viewer

A tiny local web tool to browse LeRobot (aloha-agilex) datasets: **three camera views in
sync + the 14-D action curve**, step frame-by-frame, and pick / export keyframes.

Videos are AV1, which browsers can't seek frame-accurately, so the server decodes each
requested frame server-side by **global frame index** and returns a JPEG. Episode frame
ranges come from the data parquet's global `index` column (there is no `episodes.parquet`).

## Run

```bash
/home/oem/miniconda3/envs/evo_depth/bin/python serve.py      # needs pandas + imageio[pyav] + Pillow
# then open http://localhost:8000   (PORT=xxxx to change)
```

1. Type a dataset dir in the box, e.g. `/data/datasets/piper/block`, click **Load**.
2. Pick an episode; scrub the slider or `←`/`→` (`shift` = ±10). The three cameras and the
   action-curve cursor stay in sync; **click the curve to jump** to that frame.
3. `m` / **＋ Mark** to collect keyframes.
4. Set an output dir and **Save frames** → writes `f<idx>_<cam>.png` for every camera plus
   `picks.json` (idx / episode / action per pick). **Export list** shows a pastable `PICKS = {...}`.

If a system proxy (e.g. Clash) intercepts localhost and the page/frames don't load, add
`localhost, 127.0.0.1` to the proxy's bypass list (or disable it while using the viewer).

## Shortcuts
`←`/`→` step · `shift`+arrow jump 10 · `m` mark · `[` / `]` prev/next episode.

## Endpoints (for scripting)
- `GET /load?path=<dir>` → manifest (cameras, fps, episodes `{i,start,end,n,language}`, dims)
- `GET /frame?path=&cam=&idx=` → JPEG of that global frame
- `GET /series?path=&ep=` → `{start, action:[T×D], state:[T×D]}`
- `POST /save` `{dataset_path, out_dir, picks:[idx…]}` → PNGs + picks.json
