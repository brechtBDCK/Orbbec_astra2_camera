# Orbbec Astra2 Camera Runner

Minimal Python runner for Orbbec Astra2 using `pyorbbecsdk`. It supports live depth+color preview, optional depth filters, and RGB‑D + point cloud capture (including undistorted versions).

## Quick start

1) Activate venv (if you have one):
```
source venv_orbbec/bin/activate
```

2) Install dependencies (venv optional but recommended):
```
pip install -r requirements.txt
```

3) Run:
```
python main.py
```

## Configuration

Edit `config.toml` at the repo root. The main categories are:
- `run` (offline info printing)
- `streams` (depth/color profile selection)
- `filters` (depth post‑processing)
- `display` (preview + warmup)
- `capture` (output folders + calibration JSON)

## Key config knobs

- Warmup: `display.warmup_frames` (skip initial frames; helps temporal stability)
- Depth format: `streams.depth.format = "Y16"` (recommended for filters)
- Filters: `filters.temporal`, `filters.spatial`, `filters.hole_filling`, `filters.threshold`, etc.
- Captures: press `s` to save color, depth, pointcloud, and undistorted versions into `captures/`
- Capture naming: `capture_000001_*` (auto‑incremented by scanning existing files)

## Programmatic capture

If you want to trigger a capture from code (e.g., robot scripts), use:
```
from orbbec.take_images import take_images

result = take_images()  # auto‑increments capture index
print(result)
```

## Utilities

Useful helper scripts:
- `orbbec/utils/get_intrinsics_extrinsics.py` to dump intrinsics + extrinsics JSON
- `orbbec/utils/undistort_single_images.py` to undistort one color/depth pair
- `orbbec/utils/test_filters.py` to test filters in isolation
- `orbbec/utils/eye_hand_calibration.py` is a planning scratchpad for robot calibration

## Notes / troubleshooting

- If you see “unsupported format: RLE” from filters, force Y16 via `streams.depth.format = "Y16"` or pick a Y16 profile index.
- Temporal filtering improves stability but needs a few frames to settle; keep `display.warmup_frames` > 0.
