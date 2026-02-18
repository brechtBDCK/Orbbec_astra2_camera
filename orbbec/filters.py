"""Depth filter setup and application.

Filters are optional and can smooth, fill, or clamp depth values.
"""

from dataclasses import dataclass
from typing import Iterable

from pyorbbecsdk import (
    DecimationFilter,
    DisparityTransform,
    HoleFillingFilter,
    OBHoleFillingMode,
    OBSpatialAdvancedFilterParams,
    Pipeline,
    SpatialAdvancedFilter,
    TemporalFilter,
    ThresholdFilter,
)


@dataclass
class DepthFilterBundle:
    filters: list
    description: list[str]


def _parse_hole_filling_mode(value):
    if isinstance(value, OBHoleFillingMode):
        return value
    if isinstance(value, str):
        name = value.strip().upper()
        if hasattr(OBHoleFillingMode, name):
            return getattr(OBHoleFillingMode, name)
    if isinstance(value, int):
        try:
            return OBHoleFillingMode(value)
        except Exception:
            return None
    return None


def build_depth_filters(pipeline: Pipeline, filters_cfg: dict) -> DepthFilterBundle:
    """Build depth filters from config flags."""
    filters_cfg = filters_cfg or {}
    decimation_cfg = filters_cfg.get("decimation", {})
    disparity_cfg = filters_cfg.get("disparity", {})
    spatial_cfg = filters_cfg.get("spatial", {})
    temporal_cfg = filters_cfg.get("temporal", {})
    hole_cfg = filters_cfg.get("hole_filling", {})
    threshold_cfg = filters_cfg.get("threshold", {})
    filters: list = []
    description: list[str] = []

    if decimation_cfg.get("enabled", False):
        df = DecimationFilter()
        scale_note = ""
        scale_value = decimation_cfg.get("scale_value")
        if scale_value is not None:
            try:
                scale_value = int(scale_value)
                df.set_scale_value(scale_value)
                scale_note = f" (scale={scale_value})"
            except Exception:
                pass
        filters.append(df)
        description.append(f"DecimationFilter{scale_note}")
    if disparity_cfg.get("enabled", True):
        filters.append(DisparityTransform())
        description.append("DisparityTransform")
    if temporal_cfg.get("enabled", True):
        tf = TemporalFilter()
        tf.set_diff_scale(float(temporal_cfg.get("diff_scale", 0.1)))
        tf.set_weight(float(temporal_cfg.get("weight", 0.4)))
        filters.append(tf)
        description.append("TemporalFilter")
    if spatial_cfg.get("enabled", False):
        sf = SpatialAdvancedFilter()
        params = OBSpatialAdvancedFilterParams()
        params.alpha = float(spatial_cfg.get("alpha", 0.5))
        params.disp_diff = int(spatial_cfg.get("diff_threshold", 160))
        params.magnitude = int(spatial_cfg.get("magnitude", 1))
        params.radius = int(spatial_cfg.get("radius", 1))
        sf.set_filter_params(params)
        filters.append(sf)
        description.append("SpatialAdvancedFilter")
    if hole_cfg.get("enabled", False):
        hf = HoleFillingFilter()
        mode = _parse_hole_filling_mode(hole_cfg.get("mode", "NEAREST"))
        if mode is not None:
            try:
                hf.set_filling_mode(mode)
            except Exception:
                pass
        filters.append(hf)
        description.append("HoleFillingFilter")

    if threshold_cfg.get("enabled", True):
        t_min = threshold_cfg.get("min")
        t_max = threshold_cfg.get("max")
        t_min = 0 if t_min is None else int(t_min)
        t_max = 65535 if t_max is None else int(t_max)
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
