import pyorbbecsdk

OBSensorType = pyorbbecsdk.OBSensorType
Pipeline = pyorbbecsdk.Pipeline


def format_name(fmt) -> str:
    return getattr(fmt, "name", str(fmt).split(".")[-1])


def print_profiles(pipeline, sensor_type, label: str):
    profiles = pipeline.get_stream_profile_list(sensor_type)
    print(f"{label} profiles:")
    for index in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(index)
        if profile is None or not profile.is_video_stream_profile():
            continue
        profile = profile.as_video_stream_profile()
        print(
            f"  [{index:02d}] "
            f"{profile.get_width()}x{profile.get_height()}@{profile.get_fps()} "
            f"{format_name(profile.get_format())}"
        )
    print()

def print_device_info(pipeline: Pipeline) -> None:
    try:
        info = pipeline.get_device().get_device_info()
        print("\nDevice:")
        print(f"  name: {info.get_name()}")
        print(f"  pid:  {info.get_pid()}  vid: {info.get_vid()}")
        print(f"  sn:   {info.get_serial_number()}")
        print(f"  fw:   {info.get_firmware_version()}")
    except Exception as exc:
        print(f"Device info unavailable: {exc}")

def main():
    pipeline = Pipeline()
    print_profiles(pipeline, OBSensorType.DEPTH_SENSOR, "Depth")
    print_profiles(pipeline, OBSensorType.COLOR_SENSOR, "Color")
    print_device_info(pipeline)


if __name__ == "__main__":
    main()
