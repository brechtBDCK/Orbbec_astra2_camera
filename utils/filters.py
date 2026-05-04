import pyorbbecsdk as sdk

DecimationFilter = sdk.DecimationFilter
DisparityTransform = sdk.DisparityTransform
HoleFillingFilter = sdk.HoleFillingFilter
OBHoleFillingMode = sdk.OBHoleFillingMode
OBSpatialAdvancedFilterParams = sdk.OBSpatialAdvancedFilterParams
SpatialAdvancedFilter = sdk.SpatialAdvancedFilter
TemporalFilter = sdk.TemporalFilter
ThresholdFilter = sdk.ThresholdFilter

def build_depth_filter_pipeline(filters_cfg: dict | None) -> dict:
    filters_cfg = filters_cfg or {}

    decimation_cfg = filters_cfg.get("decimation", {})
    disparity_cfg = filters_cfg.get("disparity", {})
    spatial_cfg = filters_cfg.get("spatial", {})
    temporal_cfg = filters_cfg.get("temporal", {})
    hole_cfg = filters_cfg.get("hole_filling", {})
    threshold_cfg = filters_cfg.get("threshold", {})

    decimation_scale = int(decimation_cfg.get("scale_value", 0))
    spatial_alpha = float(spatial_cfg.get("alpha", 0.5))
    spatial_diff = int(spatial_cfg.get("diff_threshold", 160))
    spatial_magnitude = int(spatial_cfg.get("magnitude", 1))
    spatial_radius = int(spatial_cfg.get("radius", 1))
    temporal_diff = float(temporal_cfg.get("diff_scale", 0.1))
    temporal_weight = float(temporal_cfg.get("weight", 0.4))
    hole_filling_mode = hole_cfg.get("mode", "NEAREST")
    threshold_min = int(threshold_cfg.get("min", 0))
    threshold_max = int(threshold_cfg.get("max", 20000))

    filters = []
    description = []

    if decimation_cfg.get("enabled", False):
        filter_obj = DecimationFilter()
        if decimation_scale > 0:
            filter_obj.set_scale_value(decimation_scale)
        filters.append(filter_obj)
        description.append(f"DecimationFilter(scale={decimation_scale or 'default'})")

    if disparity_cfg.get("enabled", False):
        filters.append(DisparityTransform())
        description.append("DisparityTransform(depth->disp)")

    if spatial_cfg.get("enabled", False):
        filter_obj = SpatialAdvancedFilter()
        params = OBSpatialAdvancedFilterParams()
        params.alpha = spatial_alpha
        params.disp_diff = spatial_diff
        params.magnitude = spatial_magnitude
        params.radius = spatial_radius
        filter_obj.set_filter_params(params)
        filters.append(filter_obj)
        description.append(
            f"SpatialAdvancedFilter(alpha={spatial_alpha:g}, diff={spatial_diff}, magnitude={spatial_magnitude}, radius={spatial_radius})"
        )

    if temporal_cfg.get("enabled", False):
        filter_obj = TemporalFilter()
        filter_obj.set_diff_scale(temporal_diff)
        filter_obj.set_weight(temporal_weight)
        filters.append(filter_obj)
        description.append(
            f"TemporalFilter(diff_scale={temporal_diff:g}, weight={temporal_weight:g})"
        )

    if disparity_cfg.get("enabled", False):
        filters.append(DisparityTransform())
        description.append("DisparityTransform(disp->depth)")

    if hole_cfg.get("enabled", False):
        filter_obj = HoleFillingFilter()
        if hole_filling_mode is not None:
            filter_obj.set_filling_mode(hole_filling_mode)
        filters.append(filter_obj)
        description.append(f"HoleFillingFilter(mode={hole_filling_mode})")

    if threshold_cfg.get("enabled", False):
        filter_obj = ThresholdFilter()
        ok = filter_obj.set_value_range(threshold_min, threshold_max)
        filters.append(filter_obj)
        description.append(f"ThresholdFilter({threshold_min}-{threshold_max} mm, ok={ok})")

    info_lines = [
        f"Decimation: {'on' if decimation_cfg.get('enabled', False) else 'off'} (scale={decimation_scale})",
        f"Disparity: {'on' if disparity_cfg.get('enabled', False) else 'off'}",
        f"Spatial: {'on' if spatial_cfg.get('enabled', False) else 'off'} (a={spatial_alpha:g} diff={spatial_diff} mag={spatial_magnitude} r={spatial_radius})",
        f"Temporal: {'on' if temporal_cfg.get('enabled', False) else 'off'} (diff={temporal_diff:g} w={temporal_weight:g})",
        f"Hole fill: {'on' if hole_cfg.get('enabled', False) else 'off'} ({hole_filling_mode})",
        f"Threshold: {'on' if threshold_cfg.get('enabled', False) else 'off'} ({threshold_min}-{threshold_max} mm)",
    ]

    return {
        "filters": filters,
        "description": description,
        "info_lines": info_lines,
    }


def apply_depth_filters(depth_frame, filters):
    current = depth_frame
    if current is None:
        return None

    for filter_obj in filters:
        try:
            if hasattr(filter_obj, "is_enabled") and not filter_obj.is_enabled():
                continue
        except Exception:
            pass

        try:
            new_frame = filter_obj.process(current)
        except Exception:
            continue
        if new_frame is None:
            continue

        try:
            current = new_frame.as_depth_frame()
        except Exception:
            current = new_frame

    return current
