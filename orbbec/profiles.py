"""Stream profile helpers.

Handles listing and selection of depth/color profiles.
Profiles define resolution, FPS, and format.
"""

from pyorbbecsdk import OBFormat, OBSensorType, Pipeline


def _format_name(fmt: OBFormat) -> str:
    """Best-effort display name for formats."""
    try:
        return fmt.name
    except Exception:
        return str(fmt)


def parse_ob_format(name: str | None, default: OBFormat) -> OBFormat:
    """Parse an OBFormat enum name or return the default."""
    if not name:
        return default
    normalized = name.strip().upper().replace("-", "_")
    fmt = getattr(OBFormat, normalized, None)
    if fmt is None:
        raise ValueError(
            f"Unknown format '{name}'. Try values like Y16, RGB, BGR, MJPG, NV12."
        )
    return fmt


def get_stream_profile_list(pipeline: Pipeline, sensor_type: OBSensorType):
    """Fetch available profiles for a given sensor type."""
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    if profile_list is None:
        raise RuntimeError(f"No stream profiles available for {sensor_type}.")
    return profile_list


def list_video_profiles(pipeline: Pipeline, sensor_type: OBSensorType, label: str) -> None:
    """Print profiles with index so you can pick a PROFILE_INDEX in config."""
    try:
        profile_list = get_stream_profile_list(pipeline, sensor_type)
    except Exception as exc:
        print(f"{label}: unavailable ({exc})")
        return

    count = profile_list.get_count()
    if count == 0:
        print(f"{label}: no profiles found")
        return

    print(f"\n{label} profiles:")
    for i in range(count):
        sp = profile_list.get_stream_profile_by_index(i)
        if sp is None or not sp.is_video_stream_profile():
            continue
        vsp = sp.as_video_stream_profile()
        width = vsp.get_width()
        height = vsp.get_height()
        fps = vsp.get_fps()
        fmt = _format_name(sp.get_format())
        print(f"  [{i:02d}] {width}x{height}@{fps} {fmt}")


def choose_video_profile(
    pipeline: Pipeline,
    sensor_type: OBSensorType,
    *,
    index: int | None,
    width: int | None,
    height: int | None,
    fps: int | None,
    format_name: str | None,
    default_format: OBFormat,
):
    """Choose a video profile by index, by params, or by device default.

    Note: If index is set, that profile already defines width/height/fps.
    """
    profile_list = get_stream_profile_list(pipeline, sensor_type)

    if index is not None:
        sp = profile_list.get_stream_profile_by_index(index)
        if sp is None or not sp.is_video_stream_profile():
            raise RuntimeError(f"Profile index {index} is not a video profile.")
        return sp.as_video_stream_profile()

    explicit = any(v is not None for v in (width, height, fps, format_name))
    if explicit:
        w = 0 if width is None else int(width)
        h = 0 if height is None else int(height)
        f = 0 if fps is None else int(fps)
        fmt = parse_ob_format(format_name, default_format)
        profile = profile_list.get_video_stream_profile(w, h, fmt, f)
        if profile is None:
            raise RuntimeError(
                f"No matching profile for {sensor_type} with "
                f"{w}x{h}@{f} {_format_name(fmt)}."
            )
        return profile

    profile = profile_list.get_default_video_stream_profile()
    if profile is None:
        raise RuntimeError(f"No default video profile for {sensor_type}.")
    return profile
