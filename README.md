# Orbbec Astra2 Camera Runner

Minimal Python runner for Orbbec Astra2 using `pyorbbecsdk`. It supports live depth+color preview and depth-only point cloud capture with optional depth filters.

## Quick start

1) Activate venv (if you have one):
```
source venv_orbbec/bin/activate
```

2) Run:
```
python main_run/main.py
```

## Main modes

Configure in `main_run/config.py`:
- `MODE = "depth_with_color"` -> live depth + color preview
- `MODE = "pointcloud_with_color"` -> depth-only point clouds (saved as PLY)

## Key config knobs

- Warmup: `WARMUP_FRAMES` (skip initial frames; helps temporal stability)
- Depth format: `DEPTH_FORMAT_NAME = "Y16"` (recommended for filters)
- Filters: `ENABLE_TEMPORAL`, `ENABLE_SPATIAL`, `ENABLE_HOLE_FILLING`, `ENABLE_THRESHOLD`, etc.
- Point clouds: `POINTCLOUD_AVERAGE_N_FRAMES`, `POINTCLOUD_SAVE_PLY`, `POINTCLOUD_SAVE_DIR`

## Utilities

- `get_intrinsics_extrinsics.py` prints intrinsics and extrinsics for the default profiles.
- `test_filters.py` is a standalone depth filter demo.

## Notes / troubleshooting

- If you see “unsupported format: RLE” from filters, force Y16 with `DEPTH_FORMAT_NAME = "Y16"` or pick a Y16 profile via `DEPTH_PROFILE_INDEX`.
- Temporal filtering improves stability but needs a few frames to settle; keep `WARMUP_FRAMES` > 0.

