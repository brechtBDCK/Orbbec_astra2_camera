# Orbbec Astra2 Notes

## 1) Recommended depth filters (from post_processing.py)
These are the per-device recommended filters reported by the SDK and their default enabled states:
- DecimationFilter: disabled
- SpatialAdvancedFilter: disabled
- TemporalFilter: disabled
- HoleFillingFilter: disabled
- DisparityTransform: enabled
- ThresholdFilter: disabled

## 2) SDK filter registry (from FilterFactory log)
The SDK reports a larger list of filter types that exist in the runtime. This is a global registry, not necessarily all active for Astra2:
- Align
- ConfidenceEstimator
- DecimationFilter
- DepthResize
- DisparityTransform
- EdgeNoiseRemovalFilter
- FormatConverter
- FrameFlip
- FrameMirror
- FrameRotate
- FrameUnpacker
- G2PixelOffsetFix
- G2XLDecompress
- HDRMerge
- HardwareD2DCorrectionFilter
- HoleFillingFilter
- IMUCorrector
- IMUFrameReversion
- NoiseRemovalFilter
- OpenNIDisparityTransform
- PixelValueOffset
- PixelValueScaler
- PointCloudFilter
- SequenceIdFilter
- SpatialAdvancedFilter
- SpatialFastFilter
- SpatialModerateFilter
- TemporalFilter
- ThresholdFilter

## 3) Calibration / camera parameters (captured output)
- Depth intrinsics:
  - fx=1426.448608, fy=1426.377930
  - cx=795.896362, cy=588.158936
  - size=1600 x 1200
- Depth distortion:
  - k1..k6=0, p1=0, p2=0
- Color intrinsics:
  - fx=833.940063, fy=834.184814
  - cx=647.649292, cy=355.568390
  - size=1280 x 720
- Color distortion:
  - k1=0.113170, k2=-0.317741, k3=0.248096
  - p1=-0.000301, p2=0.000171
- Extrinsic (depth -> color):
  - rot = [0.999986, -0.00518737, -0.000331438,
           0.00518617, 0.99998,  -0.00351596,
           0.00034967, 0.0035142, 0.999994]
  - transform = [-18.6542, -0.232799, -2.25739]

## 4) Commands used (for reference)
- Example that printed the recommended filters:
  - python pyorbbecsdk/examples/post_processing.py


## 5) Documentation (for reference)
https://doc.orbbec.com/documentation/Orbbec%20Gemini%20330%20Series%20Documentation/Use%20Depth%20Post-processing%20Blocks
