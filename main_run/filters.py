"""Depth filter setup and application.

Filters are optional and can smooth, fill, or clamp depth values.
"""

from dataclasses import dataclass
from typing import Iterable

from pyorbbecsdk import (
    DecimationFilter,
    HoleFillingFilter,
    OBSensorType,
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

    if cfg.USE_RECOMMENDED_FILTERS:
        device = pipeline.get_device()
        depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
        recommended = list(depth_sensor.get_recommended_filters())
        for f in recommended:
            filters.append(f)
            try:
                description.append(f"{f.get_name()} (enabled={f.is_enabled()})")
            except Exception:
                description.append(str(f))
        return DepthFilterBundle(filters, description)

    if cfg.ENABLE_DECIMATION:
        filters.append(DecimationFilter())
        description.append("DecimationFilter")
    if cfg.ENABLE_TEMPORAL:
        filters.append(TemporalFilter())
        description.append("TemporalFilter")
    if cfg.ENABLE_SPATIAL:
        filters.append(SpatialAdvancedFilter())
        description.append("SpatialAdvancedFilter")
    if cfg.ENABLE_HOLE_FILLING:
        filters.append(HoleFillingFilter())
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
