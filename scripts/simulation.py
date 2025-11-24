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

from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
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
        
        # Simulation parameters
        self.publish_rate = rospy.Rate(30)  # 30 Hz for sensor data
        self.last_publish_time = rospy.Time.now()

 
        # Initialize PyBullet
        self._init_pybullet()
        
        # Load simulation environment
        self._load_environment()
        
        # Initialize robot
        self._load_robot()
        
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
            # Convert to uint8 for mono8 encoding (may lose precision for high object IDs)
            seg_array_uint8 = np.clip(seg_array, 0, 255).astype(np.uint8)
            seg_msg = self.bridge.cv2_to_imgmsg(seg_array_uint8, "mono8")
            seg_msg.header = header
            self.seg_pub.publish(seg_msg)
        except Exception as e:
            rospy.logwarn(f"Error publishing segmented image: {e}")
    
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
                ranges.append(self.lidar_max_range)
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
                    ranges.append(self.lidar_max_range)
        
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

