from pyorbbecsdk import Pipeline, OBSensorType

pipeline = Pipeline()

# Pick the profiles you actually intend to stream
color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

color_profile = color_list.get_default_video_stream_profile()
depth_profile = depth_list.get_default_video_stream_profile()

# Intrinsics + distortion (per-profile)
depth_intr = depth_profile.get_intrinsic()
depth_dist = depth_profile.get_distortion()

color_intr = color_profile.get_intrinsic()
color_dist = color_profile.get_distortion()

# Extrinsics (depth -> color)
extr = depth_profile.get_extrinsic_to(color_profile)

print("DEPTH intr:", depth_intr)
print("DEPTH dist:", depth_dist)
print("COLOR intr:", color_intr)
print("COLOR dist:", color_dist)
print("Extr depth->color:", extr)

# ---------------------------------------------------------------------------- #
