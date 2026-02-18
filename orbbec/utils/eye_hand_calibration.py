"""
Eye-in-hand calibration plan (camera mounted on robot end-effector).

Goal:
  - Solve the fixed transform between the end-effector (E) and camera (C).
  - Choose output convention and keep it consistent:
      * T_E_C: camera pose in end-effector frame (preferred)
      * T_C_E: end-effector pose in camera frame (inverse of T_E_C)

Frames:
  - B: robot base frame
  - E: end-effector (tool/flange) frame
  - C: camera optical frame
  - T: calibration target frame (checkerboard/ArUco)

Prereqs:
  1) Camera intrinsics + distortion (from camera calibration).
  2) Rigid calibration target with known geometry.
  3) Robot forward kinematics available to produce T_B_E for each pose.

Data collection (N poses, typically 15-30, varied orientations/positions):
  For each pose i:
    - Record robot FK: T_B_E_i.
    - Capture image from the camera.
    - Detect target pose in camera: T_C_T_i (PnP from target corners).
    - Store the pair (T_B_E_i, T_C_T_i).

Hand-eye solve (A_i * X = X * B_i):
  If target is static in the world:
    - A_i = (T_B_E_i)^-1 * T_B_E_{i+1}
    - B_i = T_C_T_i * (T_C_T_{i+1})^-1
    - Solve for X = T_E_C (or X = T_C_E if you prefer the inverse).
  Use a library solver (OpenCV calibrateHandEye or equivalent).

Validation:
  - Reconstruct target pose in base:
      T_B_T_i = T_B_E_i * T_E_C * T_C_T_i
  - Check consistency across all i (small spread in position/rotation).

FAQ / notes:
  - How do I get my "distortion"?
      * Run a camera intrinsics calibration (checkerboard/ArUco) to estimate
        fx, fy, cx, cy and distortion coefficients. Save them and reuse.
      * For Orbbec, you may also be able to read intrinsics/distortion from
        the SDK if it exposes them for the active stream.
  - What is the best rigid calibration target?
      * A high-quality printed checkerboard or an ArUco/Charuco board mounted
        to a flat, stiff backing (glass/acrylic/metal). Charuco is robust.
  - Is the camera capture image I need an RGB one?
      * Use the stream that can reliably detect the target. RGB is typical.
        Depth-only can work if you can detect the pattern, but RGB is easier.
  - Do I need to turn off all the filters?
      * For PnP pose estimation, disable filters that alter geometry or
        reproject points (spatial/temporal/decimation) on the stream used to
        detect the target. For RGB detection, depth filters don't matter.
  - What is PnP? How to detect target pose?
      * PnP (Perspective-n-Point) estimates a camera-to-target pose from 2D
        image points and known 3D target points (OpenCV solvePnP).
      * Detect checkerboard corners or ArUco/Charuco corners, match them to
        known 3D coordinates on the board, then run solvePnP.

Output:
  - Save T_E_C (4x4 homogeneous matrix) in a config file (JSON/YAML).
  - Record units (meters vs mm) and frame conventions.
"""
from fanuc_rmi import RobotClient
from orbbec.take_images import take_images

# create client (no network calls yet)
robot = RobotClient(
    host="192.168.1.22",
    startup_port=16001,
    main_port=16002,
    connect_timeout=5.0,
    socket_timeout=100.0,
    reader_timeout=100.0,
    attempts=5,
    retry_delay=0.5,
    startup_pause=0.25,
)

robot.connect()            # returns None
robot.initialize(uframe=0, utool=1)  # returns None
robot.speed_override(50)  # returns None

# move to absolute joint angles (deg)
absolute_joints = {"J1": 12.648, "J2": 59.326, "J3": -2.001, "J4": 2.868, "J5": -133.750, "J6": 103.221, "J7": 0.000, "J8": 0.000, "J9": 0.000}
robot.joint_absolute(absolute_joints, speed_percentage=40, sequence_id=1)  # returns None
take_images()  # camera takes images and saves the images
robot.wait_time(0.5, sequence_id=2)  # returns None, waits for 0.5 second

absolute_joints = {"J1": 63.252, "J2": 31.488, "J3": -35.602, "J4": 18.504, "J5": -101.313, "J6": 108.650, "J7": 0.000, "J8": 0.000, "J9": 0.000}
robot.joint_absolute(absolute_joints, speed_percentage=40, sequence_id=3)  # returns None
take_images() 
robot.wait_time(0.5, sequence_id=4)  

absolute_joints = {"J1": -50.296, "J2": 35.193, "J3": -13.797, "J4": -29.754, "J5": -117.368, "J6": 224.915, "J7": 0.000, "J8": 0.000, "J9": 0.000}
robot.joint_absolute(absolute_joints, speed_percentage=40, sequence_id=5)  # returns None
take_images()  
robot.wait_time(0.5, sequence_id=6)  

absolute_joints = {"J1": 1.534, "J2": -70.678, "J3": 31.539, "J4": -163.759, "J5": 76.388, "J6": 224.917, "J7": 0.000, "J8": 0.000, "J9": 0.000}
robot.joint_absolute(absolute_joints, speed_percentage=40, sequence_id=7)  # returns None
take_images() 
robot.wait_time(0.5, sequence_id=8)  


robot.close()  # returns None
