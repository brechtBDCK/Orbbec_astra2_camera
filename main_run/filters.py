"""Depth filter setup and application.

Filters are optional and can smooth, fill, or clamp depth values.
"""

from dataclasses import dataclass
from typing import Iterable

from pyorbbecsdk import (
    DecimationFilter,
    DisparityTransform,
    HoleFillingFilter,
    OBSensorType,
    OBSpatialAdvancedFilterParams,
    Pipeline,
    SpatialAdvancedFilter,
    TemporalFilter,
    ThresholdFilter,
)

import main_run.config as cfg


@dataclass
class DepthFilterBundle:
    filters: list
    description: list[str]


def build_depth_filters(pipeline: Pipeline) -> DepthFilterBundle:
    """Build depth filters from config flags."""
    filters: list = []
    description: list[str] = []

    if cfg.ENABLE_DECIMATION:
        df = DecimationFilter()
        scale_note = ""
        if cfg.DECIMATION_SCALE_VALUE is not None:
            try:
                scale_value = int(cfg.DECIMATION_SCALE_VALUE)
                df.set_scale_value(scale_value)
                scale_note = f" (scale={scale_value})"
            except Exception:
                pass
        filters.append(df)
        description.append(f"DecimationFilter{scale_note}")
    if cfg.ENABLE_DISPARITY:
        filters.append(DisparityTransform())
        description.append("DisparityTransform")
    if cfg.ENABLE_TEMPORAL:
        tf = TemporalFilter()
        tf.set_diff_scale(float(cfg.TEMPORAL_DIFF_SCALE))
        tf.set_weight(float(cfg.TEMPORAL_WEIGHT))
        filters.append(tf)
        description.append("TemporalFilter")
    if cfg.ENABLE_SPATIAL:
        sf = SpatialAdvancedFilter()
        params = OBSpatialAdvancedFilterParams()
        params.alpha = float(cfg.SPATIAL_ALPHA)
        params.disp_diff = int(cfg.SPATIAL_DIFF_THRESHOLD)
        params.magnitude = int(cfg.SPATIAL_MAGNITUDE)
        params.radius = int(cfg.SPATIAL_RADIUS)
        sf.set_filter_params(params)
        filters.append(sf)
        description.append("SpatialAdvancedFilter")
    if cfg.ENABLE_HOLE_FILLING:
        hf = HoleFillingFilter()
        try:
            hf.set_filling_mode(cfg.HOLE_FILLING_MODE)
        except Exception:
            pass
        filters.append(hf)
        description.append("HoleFillingFilter")

    if cfg.THRESHOLD_MIN is not None or cfg.THRESHOLD_MAX is not None:
        t_min = 0 if cfg.THRESHOLD_MIN is None else int(cfg.THRESHOLD_MIN)
        t_max = 65535 if cfg.THRESHOLD_MAX is None else int(cfg.THRESHOLD_MAX)
        thresh = ThresholdFilter()
        ok = thresh.set_value_range(t_min, t_max)
        description.append(f"ThresholdFilter[{t_min}, {t_max}] (ok={ok})")
        filters.append(thresh)

    return DepthFilterBundle(filters, description)


def apply_depth_filters(depth_frame, filters: Iterable):
    """Apply SDK depth filters in sequence, skipping disabled ones."""
    current = depth_frame
    for f in filters:
        try:
            if hasattr(f, "is_enabled") and not f.is_enabled():
                continue
        except Exception:
            pass
        try:
            new_frame = f.process(current)
        except Exception:
            continue
        if new_frame is None:
            continue
        try:
            current = new_frame.as_depth_frame()
        except Exception:
            current = new_frame
    return current
