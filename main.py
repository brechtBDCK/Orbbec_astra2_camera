from pathlib import Path
import tomllib
import pyorbbecsdk as sdk
import cv2
from utils.visuals import draw_info_panel, make_preview
from utils.convert_frame_format import color_frame_to_bgr, depth_frame_to_bgr
from utils.filters import apply_depth_filters, build_depth_filter_pipeline


def load_config(config_path: Path ) -> dict:
    path = config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)




Config = sdk.Config
OBSensorType = sdk.OBSensorType
Pipeline = sdk.Pipeline

WINDOW_NAME = "Astra2 Preview"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CONFIG_PATH = Path("./config.toml")


def profile_to_text(profile) -> str:
    fmt = profile.get_format()
    fmt_name = getattr(fmt, "name", str(fmt).split(".")[-1])
    return f"{profile.get_width()}x{profile.get_height()}@{profile.get_fps()} {fmt_name}"


def build_pipeline(cfg:dict):
    pipeline = Pipeline()
    config = Config()

    color_cfg = cfg.get("stream", {}).get("color", {})
    depth_cfg = cfg.get("stream", {}).get("depth", {})

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    color_profile_index = int(color_cfg.get("profile_index", -1))
    depth_profile_index = int(depth_cfg.get("profile_index", -1))

    if color_profile_index >= 0:
        color_profile = color_profiles.get_stream_profile_by_index(color_profile_index).as_video_stream_profile()
    else:
        color_profile = color_profiles.get_default_video_stream_profile()

    if depth_profile_index >= 0:
        depth_profile = depth_profiles.get_stream_profile_by_index(depth_profile_index).as_video_stream_profile()
    else:
        depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    return pipeline, config, profile_to_text(color_profile), profile_to_text(depth_profile)

def main():
    cfg = load_config(CONFIG_PATH)
    pipeline, config, color_profile_text, depth_profile_text = build_pipeline(cfg = cfg)
    depth_filters = build_depth_filter_pipeline(cfg.get("filters", {}))
    pipeline.start(config)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    print("Preview running. Press q or ESC to quit.")
    last_color_image = None
    last_depth_image = None

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_image = color_frame_to_bgr(frames.get_color_frame())
            depth_frame = apply_depth_filters(frames.get_depth_frame(), depth_filters["filters"])
            depth_image = depth_frame_to_bgr(depth_frame)
            if color_image is not None:
                last_color_image = color_image
            if depth_image is not None:
                last_depth_image = depth_image

            preview = make_preview(last_color_image, last_depth_image, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            if preview is None:
                continue
            preview = draw_info_panel(
                preview,
                color_profile_text=color_profile_text,
                depth_profile_text=depth_profile_text,
                filters=depth_filters["info_lines"],
            )

            cv2.imshow(WINDOW_NAME, preview) #type: ignore
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
