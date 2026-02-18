# Orbbec Astra2 Camera Runner

Minimal Python runner for Orbbec Astra2 using `pyorbbecsdk`. It supports live depth+color preview with optional depth filters and point cloud captures.

## Quick start

1) Activate venv (if you have one):
```
source venv_orbbec/bin/activate
```

2) Run:
```
python main.py
```

## Configuration

Edit `config.toml` at the repo root.

## Key config knobs

- Warmup: `display.warmup_frames` (skip initial frames; helps temporal stability)
- Depth format: `streams.depth.format = "Y16"` (recommended for filters)
- Filters: `filters.temporal`, `filters.spatial`, `filters.hole_filling`, `filters.threshold`, etc.
- Captures: press `s` to save color, depth, pointcloud, and undistorted versions into `captures/`

## Notes / troubleshooting

- If you see “unsupported format: RLE” from filters, force Y16 via `streams.depth.format = "Y16"` or pick a Y16 profile index.
- Temporal filtering improves stability but needs a few frames to settle; keep `display.warmup_frames` > 0.
