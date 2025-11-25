#!/usr/bin/env python3
"""
ROS 1 PyBullet Simulation Node for Mobile Robot with 6DOF Arm and Gripper

This node:
- Runs PyBullet simulation with the mobile robot
- Publishes RGB, depth, and segmented camera images
- Publishes 2D Lidar scan data
- Subscribes to topics for arm, gripper, and mobile base control
"""

import rospy
import pybullet
import pybullet_data
import numpy as np
import os
import cv2
import tf2_ros
import tf.transformations

from sensor_msgs.msg import Image, LaserScan, CameraInfo
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from std_msgs.msg import Header


class PyBulletSimulation:
    """Main class for PyBullet simulation with ROS integration"""
    
    def __init__(self):
        """Initialize the simulation node"""
        rospy.init_node('pybullet_simulation', anonymous=True)
        
        # Get ROS parameters
        self.use_gui = rospy.get_param('~use_gui', True)
        self.urdf_path = rospy.get_param('~urdf_path', '../urdf/mobile_robot_with_arm.urdf')
        self.time_step = 1.0 / 240.0

        self.arm_joint_positions = {
            'shoulder_pan_joint': 0.0,
            'shoulder_lift_joint': -1.257,
            'elbow_joint': 0.0,
            'wrist_1_joint': 0.0,
            'wrist_2_joint': 0.0,
            'wrist_3_joint': 0.0
        }
        self.gripper_position = 0.0
        self.cmd_vel = Twist()
        self.obstacle_ids = []  # List to store obstacle IDs
        self.wall_id = None  # Wall object ID
        
        # Simulation parameters
        self.publish_rate = rospy.Rate(30)  # 30 Hz for sensor data
        self.last_publish_time = rospy.Time.now()

 
        # Initialize PyBullet
        self._init_pybullet()
        
        # Load simulation environment
        self._load_environment()
        
        # Initialize robot
        self._load_robot()
        
        # Load obstacles around the robot
        self._load_obstacles()
        
        # Load fixed wall
        self._load_wall()
        
        # Initialize joint indices and parameters
        self._init_joint_info()
        
        # Initialize camera parameters
        self._init_camera()
        
        # Initialize lidar parameters
        self._init_lidar()
        
        # Initialize ROS publishers
        self._init_publishers()
        
        # Initialize ROS subscribers
        self._init_subscribers()
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize static transform broadcaster
        self._init_tf_broadcaster()
        
        # Control state variables
        
        rospy.loginfo("PyBullet simulation node initialized")
    
    def _init_pybullet(self):
        """Initialize PyBullet physics engine"""
        if self.use_gui:
            self.physics_client = pybullet.connect(pybullet.GUI)
            rospy.loginfo("PyBullet connected in GUI mode")
        else:
            self.physics_client = pybullet.connect(pybullet.DIRECT)
            rospy.loginfo("PyBullet connected in DIRECT mode")
        
        pybullet.resetSimulation()
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0.0, 0.0, -9.8)
        pybullet.setTimeStep(self.time_step)
    
    def _load_environment(self):
        """Load the simulation environment (floor)"""
        self.plane_id = pybullet.loadURDF("plane.urdf")
        rospy.loginfo("Loaded floor plane")
    
    def _load_robot(self):
        """Load the robot URDF"""
        car_start_pos = [0, 0, 0.1]
        car_start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        
        # Resolve URDF path
        if not os.path.isabs(self.urdf_path):
            # Relative path, resolve from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.urdf_path = os.path.join(script_dir, self.urdf_path)
        
        self.robot_id = pybullet.loadURDF(self.urdf_path, 
                                          car_start_pos, 
                                          car_start_orientation)
        rospy.loginfo(f"Loaded robot from {self.urdf_path}")
        
        # Set initial arm joint positions
        for joint_name, joint_angle in self.arm_joint_positions.items():
            joint_idx = self._get_joint_index_by_name(joint_name)
            if joint_idx is not None:
                pybullet.resetJointState(self.robot_id, joint_idx, joint_angle)
                rospy.loginfo(f"Set initial position for {joint_name}: {joint_angle:.3f} rad")
        
        # Step simulation a few times to let the arm settle
        for _ in range(10):
            pybullet.stepSimulation()
    
    def _load_obstacles(self):
        """Load obstacles (boxes) around the robot within 10m radius"""
        # Get ROS parameter for number of obstacles
        num_obstacles = rospy.get_param('~num_obstacles', 10)
        max_distance = rospy.get_param('~obstacle_max_distance', 10.0)  # meters
        min_distance = rospy.get_param('~obstacle_min_distance', 1.5)  # meters (avoid too close to robot)
        
        # Resolve box URDF path
        box_urdf_path = rospy.get_param('~box_urdf_path', '../urdf/simple_box.urdf')
        if not os.path.isabs(box_urdf_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            box_urdf_path = os.path.join(script_dir, box_urdf_path)
        
        # Box size from URDF is 0.3x0.3x0.3 meters
        box_size = 0.3
        box_half_height = box_size / 2.0  # 0.15m
        
        self.obstacle_ids = []
        
        rospy.loginfo(f"Loading {num_obstacles} obstacles within {min_distance}-{max_distance}m radius...")
        
        for i in range(num_obstacles):
            # Random distance from robot center (avoid too close)
            distance = np.random.uniform(min_distance, max_distance)
            
            # Random angle around robot (0 to 2π)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate box position (robot is at [0, 0, 0.1])
            box_x = distance * np.cos(angle)
            box_y = distance * np.sin(angle)
            # Place box so bottom sits on ground (z = 0), so position is at half height
            box_z = box_half_height  # 0.15m
            
            # Random orientation (optional - can be set to [0,0,0] for no rotation)
            box_yaw = np.random.uniform(0, 2 * np.pi)
            box_orientation = pybullet.getQuaternionFromEuler([0, 0, box_yaw])
            
            # Load the box (useFixedBase=False so boxes can fall and move due to physics)
            try:
                box_id = pybullet.loadURDF(box_urdf_path, 
                                          [box_x, box_y, box_z], 
                                          box_orientation,
                                          useFixedBase=False)
                self.obstacle_ids.append(box_id)
                rospy.loginfo(f"Obstacle {i+1} loaded at position ({box_x:.2f}, {box_y:.2f}, {box_z:.2f}), distance: {distance:.2f}m")
            except Exception as e:
                rospy.logwarn(f"Failed to load obstacle {i+1}: {e}")
        
        rospy.loginfo(f"Total {len(self.obstacle_ids)} obstacles loaded around the robot.")
        
        # Step simulation a few times to let obstacles settle
        for _ in range(20):
            pybullet.stepSimulation()
    
    def _load_wall(self):
        """Load a fixed wall (horizontal box) in the simulation"""
        # Get ROS parameters for wall configuration
        enable_wall = rospy.get_param('~enable_wall', True)
        if not enable_wall:
            rospy.loginfo("Wall loading disabled")
            return
        
        # Resolve wall URDF path
        wall_urdf_path = rospy.get_param('~wall_urdf_path', '../urdf/horizontal_box.urdf')
        if not os.path.isabs(wall_urdf_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            wall_urdf_path = os.path.join(script_dir, wall_urdf_path)
        
        # Wall position parameters (can be configured via ROS params)
        wall_x = rospy.get_param('~wall_x', 5.0)  # meters in front of robot
        wall_y = rospy.get_param('~wall_y', 0.0)  # meters to the side
        wall_z = rospy.get_param('~wall_z', 0.5)  # height (half of 1m box height)
        wall_yaw = rospy.get_param('~wall_yaw', 1.57)  # rotation around Z axis (radians)
        
        # Wall orientation (horizontal box is 10x1x1, so it's a long horizontal wall)
        # Default orientation: wall perpendicular to Y-axis (facing robot)
        wall_orientation = pybullet.getQuaternionFromEuler([0, 0, wall_yaw])
        
        try:
            # Load wall as fixed base object (won't move)
            self.wall_id = pybullet.loadURDF(wall_urdf_path,
                                            [wall_x, wall_y, wall_z],
                                            wall_orientation,
                                            useFixedBase=True)
            rospy.loginfo(f"Fixed wall loaded at position ({wall_x:.2f}, {wall_y:.2f}, {wall_z:.2f}) with yaw {wall_yaw:.2f} rad")
        except Exception as e:
            rospy.logwarn(f"Failed to load wall: {e}")
            self.wall_id = None
    
    def _get_joint_index_by_name(self, joint_name):
        """Get the joint index by searching through all joints"""
        num_joints = pybullet.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot_id, i)
            joint_name_found = joint_info[1].decode('utf-8')
            if joint_name_found == joint_name:
                return i
        return None
    
    def _get_link_index_by_name(self, link_name):
        """Get the link index by searching through all joints"""
        num_joints = pybullet.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot_id, i)
            child_link_name = joint_info[12].decode('utf-8')
            if child_link_name == link_name:
                return i
        return None
    
    def _init_joint_info(self):
        """Initialize joint indices and parameters"""
        # UR5 arm joint names
        self.arm_joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Get arm joint indices
        self.arm_joint_indices = {}
        for joint_name in self.arm_joint_names:
            joint_idx = self._get_joint_index_by_name(joint_name)
            self.arm_joint_indices[joint_name] = joint_idx
            if joint_idx is not None:
                rospy.loginfo(f"{joint_name} index: {joint_idx}")
            else:
                rospy.logwarn(f"Warning: {joint_name} not found!")
        
        # Gripper joint names
        gripper_right_joint_name = 'robotiq_2f_85_right_driver_joint'
        gripper_left_joint_name = 'robotiq_2f_85_left_driver_joint'
        
        self.gripper_right_joint_idx = self._get_joint_index_by_name(gripper_right_joint_name)
        self.gripper_left_joint_idx = self._get_joint_index_by_name(gripper_left_joint_name)
        
        if self.gripper_right_joint_idx is not None:
            rospy.loginfo(f"{gripper_right_joint_name} index: {self.gripper_right_joint_idx}")
        if self.gripper_left_joint_idx is not None:
            rospy.loginfo(f"{gripper_left_joint_name} index: {self.gripper_left_joint_idx}")
        
        # Joint effort limits
        self.joint_efforts = {
            'shoulder_pan_joint': 150.0,
            'shoulder_lift_joint': 150.0,
            'elbow_joint': 150.0,
            'wrist_1_joint': 28.0,
            'wrist_2_joint': 28.0,
            'wrist_3_joint': 28.0
        }
        self.gripper_effort = 60.0
        
        # Wheel joint names
        wheel_joint_names = [
            'left_wheel_joint',
            'right_wheel_joint',
            'front_left_wheel_joint',
            'front_right_wheel_joint'
        ]
        
        self.wheel_joint_indices = {}
        for wheel_name in wheel_joint_names:
            wheel_idx = self._get_joint_index_by_name(wheel_name)
            self.wheel_joint_indices[wheel_name] = wheel_idx
            if wheel_idx is not None:
                rospy.loginfo(f"{wheel_name} index: {wheel_idx}")
        
        self.wheel_max_torque = 200.0  # N⋅m
    
    def _init_camera(self):
        """Initialize camera parameters"""
        self.camera_fov = 60  # degrees
        self.image_width = 640
        self.image_height = 480
        self.aspect = self.image_width / self.image_height
        self.near = 0.05
        self.far = 10.0
        
        # Compute projection matrix
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(
            self.camera_fov, self.aspect, self.near, self.far
        )
        
        # Calculate camera intrinsics from FOV and image dimensions
        # FOV is in degrees, convert to radians
        fov_rad = np.radians(self.camera_fov)
        # Focal length: fx = fy = (width / 2) / tan(FOV/2)
        self.fx = self.image_width / (2.0 * np.tan(fov_rad / 2.0))
        self.fy = self.image_height / (2.0 * np.tan(fov_rad / 2.0))
        # Principal point (optical center) - typically at image center
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        
        rospy.loginfo(f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        
        # Get camera link indices
        self.rgbd_camera_link_idx = self._get_link_index_by_name("rgbd_camera_link")
        self.rgbd_camera_target_link_idx = self._get_link_index_by_name("rgbd_camera_target_vertual_link")
        
        if self.rgbd_camera_link_idx is not None:
            rospy.loginfo(f"RGB-D Camera Link Index: {self.rgbd_camera_link_idx}")
        if self.rgbd_camera_target_link_idx is not None:
            rospy.loginfo(f"RGB-D Camera Target Link Index: {self.rgbd_camera_target_link_idx}")
    
    def _init_lidar(self):
        """Initialize lidar parameters"""
        self.lidar_link_idx = self._get_link_index_by_name("lidar_link")
        if self.lidar_link_idx is not None:
            rospy.loginfo(f"Lidar Link Index: {self.lidar_link_idx}")
        
        self.lidar_num_rays = 360
        self.lidar_max_range = 10.0
    
    def _init_publishers(self):
        """Initialize ROS publishers"""
        self.rgb_pub = rospy.Publisher('camera/rgb/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('camera/depth/image_raw', Image, queue_size=1)
        self.seg_pub = rospy.Publisher('camera/segmentation/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('camera/rgb/camera_info', CameraInfo, queue_size=1)
        self.depth_camera_info_pub = rospy.Publisher('camera/depth/camera_info', CameraInfo, queue_size=1)
        self.lidar_pub = rospy.Publisher('scan', LaserScan, queue_size=1)
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=1)
    
    def _init_subscribers(self):
        """Initialize ROS subscribers"""
        # Subscribe to joint state for arm control
        rospy.Subscriber('arm/joint_states', JointState, self._arm_joint_state_callback)
        
        # Subscribe to individual joint topics (alternative control method)
        for joint_name in self.arm_joint_names:
            topic_name = f'arm/{joint_name}/command'
            rospy.Subscriber(topic_name, Float64, 
                           lambda msg, name=joint_name: self._arm_joint_command_callback(name, msg))
        
        # Subscribe to gripper control
        rospy.Subscriber('gripper/command', Float64, self._gripper_command_callback)
        
        # Subscribe to mobile base velocity commands
        rospy.Subscriber('cmd_vel', Twist, self._cmd_vel_callback)
    
    def _init_tf_broadcaster(self):
        """Initialize static transform broadcaster for camera_link to lidar_link"""
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Get transform between camera_link and lidar_link from PyBullet
        # Both are attached to base_link, so we can get their relative transform
        self._publish_camera_lidar_transform()
    
    def _arm_joint_state_callback(self, msg):
        """Callback for arm joint state control"""
        for i, name in enumerate(msg.name):
            if name in self.arm_joint_positions:
                if i < len(msg.position):
                    self.arm_joint_positions[name] = msg.position[i]
    
    def _arm_joint_command_callback(self, joint_name, msg):
        """Callback for individual arm joint command"""
        if joint_name in self.arm_joint_positions:
            self.arm_joint_positions[joint_name] = msg.data
    
    def _gripper_command_callback(self, msg):
        """Callback for gripper control"""
        # Clamp gripper position between 0.0 (closed) and 0.834 (open)
        self.gripper_position = np.clip(msg.data, 0.0, 0.834)
    
    def _cmd_vel_callback(self, msg):
        """Callback for mobile base velocity commands"""
        self.cmd_vel = msg
    
    def _control_robot(self):
        """Apply control commands to the robot"""
        # Control arm joints
        for joint_name, joint_angle in self.arm_joint_positions.items():
            joint_idx = self.arm_joint_indices.get(joint_name)
            if joint_idx is not None:
                joint_effort = self.joint_efforts.get(joint_name, 150.0)
                pybullet.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    pybullet.POSITION_CONTROL,
                    targetPosition=joint_angle,
                    force=joint_effort
                )
        
        # Control gripper
        if self.gripper_right_joint_idx is not None:
            pybullet.setJointMotorControl2(
                self.robot_id,
                self.gripper_right_joint_idx,
                pybullet.POSITION_CONTROL,
                targetPosition=self.gripper_position,
                force=self.gripper_effort
            )
        
        if self.gripper_left_joint_idx is not None:
            pybullet.setJointMotorControl2(
                self.robot_id,
                self.gripper_left_joint_idx,
                pybullet.POSITION_CONTROL,
                targetPosition=self.gripper_position,
                force=self.gripper_effort
            )
        
        # Control mobile base
        # Convert cmd_vel (linear.x, angular.z) to wheel velocities
        # Assuming wheel radius of 0.08m and base width of ~0.5m
        wheel_radius = 0.08
        base_width = 0.5
        
        linear_velocity = self.cmd_vel.linear.x
        angular_velocity = self.cmd_vel.angular.z
        
        # Convert to wheel angular velocities (rad/s)
        # For differential drive: v_left = (v - w*L/2) / r, v_right = (v + w*L/2) / r
        # where v is linear velocity, w is angular velocity, L is base width, r is wheel radius
        left_wheel_velocity = (linear_velocity + (angular_velocity * base_width / 2.0)) / wheel_radius
        right_wheel_velocity = (linear_velocity - (angular_velocity * base_width / 2.0)) / wheel_radius
        
        # Apply to all wheels
        wheel_names = ['left_wheel_joint', 'front_left_wheel_joint']
        for wheel_name in wheel_names:
            if wheel_name in self.wheel_joint_indices:
                wheel_idx = self.wheel_joint_indices[wheel_name]
                if wheel_idx is not None:
                    pybullet.setJointMotorControl2(
                        self.robot_id,
                        wheel_idx,
                        pybullet.VELOCITY_CONTROL,
                        targetVelocity=left_wheel_velocity,
                        force=self.wheel_max_torque
                    )
        
        wheel_names = ['right_wheel_joint', 'front_right_wheel_joint']
        for wheel_name in wheel_names:
            if wheel_name in self.wheel_joint_indices:
                wheel_idx = self.wheel_joint_indices[wheel_name]
                if wheel_idx is not None:
                    pybullet.setJointMotorControl2(
                        self.robot_id,
                        wheel_idx,
                        pybullet.VELOCITY_CONTROL,
                        targetVelocity=right_wheel_velocity,
                        force=self.wheel_max_torque
                    )
    
    def _get_camera_images(self):
        """Capture camera images from PyBullet"""
        # Get camera link positions
        try:
            if self.rgbd_camera_link_idx is not None:
                camera_link_state = pybullet.getLinkState(self.robot_id, self.rgbd_camera_link_idx)
                camera_link_position = camera_link_state[0]
                camera_link_orientation = camera_link_state[1]
            else:
                # Fallback: use base position + offset
                base_state = pybullet.getBasePositionAndOrientation(self.robot_id)
                camera_link_position = [
                    base_state[0][0] + 0.12, 
                    base_state[0][1], 
                    base_state[0][2] + 0.08
                ]
                camera_link_orientation = base_state[1]
            
            if self.rgbd_camera_target_link_idx is not None:
                camera_target_state = pybullet.getLinkState(self.robot_id, self.rgbd_camera_target_link_idx)
                camera_target_position = camera_target_state[0]
            else:
                # Fallback: position camera target 0.2m in front of camera
                camera_target_position = [
                    camera_link_position[0] + 0.2,
                    camera_link_position[1],
                    camera_link_position[2]
                ]
        except:
            # If link access fails, use base position with offset
            base_state = pybullet.getBasePositionAndOrientation(self.robot_id)
            camera_link_position = [
                base_state[0][0] + 0.12,
                base_state[0][1],
                base_state[0][2] + 0.08
            ]
            camera_target_position = [
                camera_link_position[0] + 0.2,
                camera_link_position[1],
                camera_link_position[2]
            ]
        
        # Compute view matrix
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=camera_link_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=[0, 0, 1]
        )
        
        # Get camera image
        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        
        return rgb_img, depth_img, seg_img
    
    def _publish_camera_images(self, rgb_img, depth_img, seg_img):
        """Publish camera images to ROS topics"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"
        
        # Publish RGB image
        try:
            # PyBullet returns RGB image as 1D array (width * height * 4) RGBA
            rgb_array = np.array(rgb_img, dtype=np.uint8)
            rgb_array = rgb_array.reshape((self.image_height, self.image_width, 4))
            # Remove alpha channel and convert to RGB
            rgb_array = rgb_array[:, :, :3]
            # PyBullet returns in RGBA format, but we need RGB
            # The array is already in the correct format, just remove alpha
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_array, "rgb8")
            rgb_msg.header = header
            self.rgb_pub.publish(rgb_msg)
        except Exception as e:
            rospy.logwarn(f"Error publishing RGB image: {e}")
        
        # Publish depth image
        try:
            # Depth image is in meters, PyBullet returns as 1D array
            # Each pixel value represents the Z-depth (distance from camera) in meters
            # Encoding: "32FC1" (32-bit float, single channel)
            # To extract depth values in another node:
            #   1. Subscribe to '/camera/depth/image_raw'
            #   2. Use cv_bridge: depth_array = bridge.imgmsg_to_cv2(msg, "32FC1")
            #   3. Access depth at pixel (u, v): depth = depth_array[v, u]  # Note: row, col order
            #   4. depth_array is numpy array of shape (height, width) with values in meters
            depth_array = np.array(depth_img, dtype=np.float32)
            depth_array = depth_array.reshape((self.image_height, self.image_width))
            depth_msg = self.bridge.cv2_to_imgmsg(depth_array, "32FC1")
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)
        except Exception as e:
            rospy.logwarn(f"Error publishing depth image: {e}")
        
        # Publish segmented image
        try:
            # Segmentation image contains object IDs
            seg_array = np.array(seg_img, dtype=np.int32)
            seg_array = seg_array.reshape((self.image_height, self.image_width))
            
            # Get unique object IDs (excluding -1 which is typically "no object")
            unique_ids = np.unique(seg_array)
            unique_ids = unique_ids[unique_ids >= 0]  # Remove negative IDs (background/no object)
            
            # Create a color mapping for each object ID
            seg_colormap = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            
            if len(unique_ids) > 0:
                # Generate distinct colors for each object ID
                # Use a hash-based approach to get consistent colors for each ID
                for obj_id in unique_ids:
                    mask = seg_array == obj_id
                    # Generate a color based on object ID using HSV color space
                    # Use object ID to generate hue (0-179 for OpenCV HSV)
                    hue = int((obj_id * 179) % 180)  # Wrap around HSV hue range
                    # Create a full saturation, full value color
                    color_hsv = np.uint8([[[hue, 255, 255]]])
                    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                    seg_colormap[mask] = color_bgr
            
            # For background/no object pixels (ID < 0 or not in unique_ids), use black
            background_mask = ~np.isin(seg_array, unique_ids)
            seg_colormap[background_mask] = [0, 0, 0]  # Black for background
            
            # Add bounding boxes for each detected object
            if len(unique_ids) > 0:
                for obj_id in unique_ids:
                    mask = seg_array == obj_id
                    # Find bounding box coordinates
                    coords = np.column_stack(np.where(mask))
                    if len(coords) > 0:
                        # Get min/max coordinates (note: y is first, x is second in np.where)
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        
                        # Get color for this object ID (same as used for the mask)
                        hue = int((obj_id * 179) % 180)
                        color_hsv = np.uint8([[[hue, 255, 255]]])
                        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                        color_tuple = tuple(map(int, color_bgr))
                        
                        # Draw bounding box (thick line for visibility)
                        cv2.rectangle(seg_colormap, (x_min, y_min), (x_max, y_max), 
                                     color_tuple, 2)
                        
                        # Add label with object ID
                        label = f"ID:{obj_id}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        # Draw background rectangle for text (slightly larger for padding)
                        cv2.rectangle(seg_colormap, (x_min, y_min - label_size[1] - 4), 
                                     (x_min + label_size[0] + 4, y_min), color_tuple, -1)
                        # Draw text in white for contrast
                        cv2.putText(seg_colormap, label, (x_min + 2, y_min - 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert BGR to RGB for ROS (OpenCV uses BGR, ROS expects RGB)
            seg_colormap_rgb = cv2.cvtColor(seg_colormap, cv2.COLOR_BGR2RGB)
            
            # Publish as RGB8 image
            seg_msg = self.bridge.cv2_to_imgmsg(seg_colormap_rgb, "rgb8")
            seg_msg.header = header
            self.seg_pub.publish(seg_msg)
        except Exception as e:
            rospy.logwarn(f"Error publishing segmented image: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
        
        # Publish camera_info for RGB and depth
        self._publish_camera_info(header)
    
    def _publish_camera_info(self, header):
        """Publish camera_info message with camera intrinsics"""
        try:
            # Create CameraInfo message
            camera_info = CameraInfo()
            camera_info.header = header
            camera_info.width = self.image_width
            camera_info.height = self.image_height
            
            # Set distortion model (no distortion for simulated camera)
            camera_info.distortion_model = "plumb_bob"
            
            # Distortion parameters (K1, K2, T1, T2, K3) - all zeros for no distortion
            camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Camera intrinsic matrix K (3x3)
            # [fx  0  cx]
            # [0  fy  cy]
            # [0   0   1]
            camera_info.K = [
                self.fx, 0.0, self.cx,
                0.0, self.fy, self.cy,
                0.0, 0.0, 1.0
            ]
            
            # Rectification matrix (identity for no rectification)
            camera_info.R = [1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0]
            
            # Projection matrix P (3x4)
            # Same as K but with extra column of zeros
            camera_info.P = [
                self.fx, 0.0, self.cx, 0.0,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            # Binning (no binning)
            camera_info.binning_x = 1
            camera_info.binning_y = 1
            
            # ROI (full image)
            camera_info.roi.x_offset = 0
            camera_info.roi.y_offset = 0
            camera_info.roi.height = self.image_height
            camera_info.roi.width = self.image_width
            camera_info.roi.do_rectify = False
            
            # Publish for both RGB and depth cameras (same intrinsics)
            self.camera_info_pub.publish(camera_info)
            
            # Update header frame_id for depth camera_info
            depth_camera_info = camera_info
            depth_camera_info.header.frame_id = "camera_link"  # Same frame
            self.depth_camera_info_pub.publish(depth_camera_info)
            
        except Exception as e:
            rospy.logwarn(f"Error publishing camera_info: {e}")
    
    def _publish_camera_lidar_transform(self):
        """Publish static transform between camera_link and lidar_link"""
        try:
            # Get link states from PyBullet to compute relative transform
            if self.rgbd_camera_link_idx is not None and self.lidar_link_idx is not None:
                camera_state = pybullet.getLinkState(self.robot_id, self.rgbd_camera_link_idx)
                lidar_state = pybullet.getLinkState(self.robot_id, self.lidar_link_idx)
                
                camera_pos = np.array(camera_state[0])
                camera_quat = camera_state[1]
                lidar_pos = np.array(lidar_state[0])
                lidar_quat = lidar_state[1]
                
                # Compute relative transform: T_camera_lidar = T_camera_world^-1 * T_lidar_world
                # Translation: relative position
                relative_pos = camera_pos - lidar_pos
                # Rotation: relative rotation
                relative_quat = pybullet.getQuaternionFromEuler([-1.57, 0.0, -1.57])
                rospy.loginfo("Using tf transformation values for camera-lidar transform")
            else:
                # Fallback: Use URDF values if links not found
                # From URDF: lidar at (0.0, 0.0, 0.0975), camera at (0.4, 0.0, 0.12) relative to base_link
                # Relative: (-0.4, 0.0, -0.0225)
                relative_pos = [-0.4, 0.0, -0.0225]
                relative_quat = [0.0, 0.0, 0.0, 1.0]  # No rotation
                rospy.logwarn("Using URDF fallback values for camera-lidar transform")
            
            # Create transform message
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "lidar_link"
            transform.child_frame_id = "camera_link"
            
            transform.transform.translation.x = relative_pos[0]
            transform.transform.translation.y = relative_pos[1]
            transform.transform.translation.z = relative_pos[2]
            
            transform.transform.rotation.x = relative_quat[0]
            transform.transform.rotation.y = relative_quat[1]
            transform.transform.rotation.z = relative_quat[2]
            transform.transform.rotation.w = relative_quat[3]
            
            # Publish static transform
            self.static_broadcaster.sendTransform(transform)
            rospy.loginfo(f"Published static transform: camera_link -> lidar_link "
                         f"translation=({relative_pos[0]:.3f}, {relative_pos[1]:.3f}, {relative_pos[2]:.3f})")
            
        except Exception as e:
            rospy.logwarn(f"Error publishing camera-lidar transform: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
    
    def _get_2d_lidar_scan(self):
        """Get 2D lidar point cloud using ray casting"""
        if self.lidar_link_idx is None:
            return None, None
        
        # Get lidar position and orientation
        lidar_state = pybullet.getLinkState(self.robot_id, self.lidar_link_idx)
        lidar_position = np.array(lidar_state[0])
        
        # Get base orientation for coordinate transformation
        base_state = pybullet.getBasePositionAndOrientation(self.robot_id)
        base_orientation = base_state[1]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(pybullet.getMatrixFromQuaternion(base_orientation)).reshape(3, 3)
        
        # Prepare rays
        angles = np.linspace(0, 2 * np.pi, self.lidar_num_rays, endpoint=False)
        ray_starts = []
        ray_ends = []
        ray_directions_world = []
        
        for angle in angles:
            # Calculate ray direction in lidar's local frame (horizontal plane, Z=0)
            ray_direction_local = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Rotate ray direction to world coordinates
            ray_direction_world = rotation_matrix @ ray_direction_local
            ray_directions_world.append(ray_direction_world)
            
            # Calculate ray end position
            ray_start = lidar_position
            ray_end = lidar_position + ray_direction_world * self.lidar_max_range
            
            ray_starts.append(ray_start)
            ray_ends.append(ray_end)
        
        # Cast all rays at once
        ray_results = pybullet.rayTestBatch(ray_starts, ray_ends)
        
        # Process results
        ranges = []
        for i, angle in enumerate(angles):
            ray_result = ray_results[i]
            hit_object_id = ray_result[0]
            hit_fraction = ray_result[2]
            hit_position = ray_result[3]
            ray_direction_world = ray_directions_world[i]
            
            if hit_object_id != -1 and hit_object_id != self.robot_id:
                # Hit an external object
                distance = np.linalg.norm(np.array(hit_position) - lidar_position)
                ranges.append(distance)
            elif hit_object_id == -1:
                # No hit, use max range
                ranges.append(self.lidar_max_range + 2)
            else:
                # Hit the robot itself, try to extend the ray
                min_distance = 0.4
                ray_start_extended = lidar_position + ray_direction_world * min_distance
                ray_end_extended = lidar_position + ray_direction_world * self.lidar_max_range
                
                ray_result_extended = pybullet.rayTest(ray_start_extended, ray_end_extended)
                
                if ray_result_extended[0][0] != -1 and ray_result_extended[0][0] != self.robot_id:
                    hit_position_extended = ray_result_extended[0][3]
                    distance = np.linalg.norm(np.array(hit_position_extended) - lidar_position)
                    ranges.append(distance)
                else:
                    ranges.append(self.lidar_max_range + 2)
        
        return np.array(ranges), angles
    
    def _publish_lidar_scan(self, ranges, angles):
        """Publish 2D lidar scan as LaserScan message"""
        if ranges is None:
            return
        
        scan_msg = LaserScan()
        scan_msg.header.stamp = rospy.Time.now()
        scan_msg.header.frame_id = "lidar_link"
        
        scan_msg.angle_min = 0.0
        scan_msg.angle_max = 2 * np.pi
        scan_msg.angle_increment = 2 * np.pi / self.lidar_num_rays
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / 30.0  # Assuming 30 Hz
        scan_msg.range_min = 0.05
        scan_msg.range_max = self.lidar_max_range
        scan_msg.ranges = ranges.tolist()
        
        self.lidar_pub.publish(scan_msg)
    
    def _publish_joint_states(self):
        """Publish joint states"""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.header.frame_id = "base_link"
        
        # Get all joint states
        num_joints = pybullet.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot_id, i)
            joint_state = pybullet.getJointState(self.robot_id, i)
            
            joint_state_msg.name.append(joint_info[1].decode('utf-8'))
            joint_state_msg.position.append(joint_state[0])
            joint_state_msg.velocity.append(joint_state[1])
            joint_state_msg.effort.append(joint_state[3])
        
        self.joint_state_pub.publish(joint_state_msg)
    
    def run(self):
        """Main simulation loop"""
        rospy.loginfo("Starting simulation loop...")
        
        while not rospy.is_shutdown():
            # Apply control commands
            self._control_robot()
            
            # Step simulation
            pybullet.stepSimulation()
            
            # Publish sensor data at ~30 Hz
            current_time = rospy.Time.now()
            if (current_time - self.last_publish_time).to_sec() >= (1.0 / 30.0):
                # Get and publish camera images
                rgb_img, depth_img, seg_img = self._get_camera_images()
                self._publish_camera_images(rgb_img, depth_img, seg_img)
                
                # Get and publish lidar scan
                ranges, angles = self._get_2d_lidar_scan()
                self._publish_lidar_scan(ranges, angles)
                
                # Publish joint states
                self._publish_joint_states()
                
                self.last_publish_time = current_time
            
            self.publish_rate.sleep()
    
    def shutdown(self):
        """Cleanup on shutdown"""
        pybullet.disconnect()
        rospy.loginfo("PyBullet simulation disconnected")


if __name__ == '__main__':
    try:
        sim = PyBulletSimulation()
        sim.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'sim' in locals():
            sim.shutdown()

