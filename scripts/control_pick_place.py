#!/usr/bin/env python3
"""
Control script for pick and place operation.

This script:
- Navigates robot to pickup location (0, 12)
- Lowers arm and picks object
- Navigates to bin location (0, -13)
- Lowers arm and releases object
"""

import rospy
import numpy as np
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64, Bool


class ControlPickPlace:
    def __init__(self):
        rospy.init_node('control_pick_place', anonymous=True)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.shoulder_lift_pub = rospy.Publisher('/arm/shoulder_lift_joint/command', Float64, queue_size=1)
        self.gripper_pick_pub = rospy.Publisher('/gripper/pick', Bool, queue_size=1)
        
        # TF listener for robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Wait for TF to be available
        rospy.sleep(1.0)
        
        # State machine states
        self.state = 'IDLE'
        self.states = {
            'IDLE': self._state_idle,
            'NAVIGATE_TO_PICKUP': self._state_navigate_to_pickup,
            'LOWER_ARM_PICKUP': self._state_lower_arm_pickup,
            'PICK_OBJECT': self.pick,
            'RAISE_ARM_PICKUP': self._state_raise_arm_pickup,
            'NAVIGATE_TO_BIN': self._state_navigate_to_bin,
            'LOWER_ARM_BIN': self._state_lower_arm_bin,
            'RELEASE_OBJECT': self.place_in_bin,
            'DONE': self._state_done
        }
        
        # Waypoints
        self.pickup_waypoint = np.array([0.0, 12.0])
        self.bin_waypoint = np.array([0.0, -13.0])
        
        # Navigation parameters
        self.angle_tolerance = 0.1  # radians
        self.max_linear_vel = 4.5  # m/s
        self.max_angular_vel = 9  # rad/s
        self.kp_linear = 1.0  # Proportional gain for linear velocity
        self.kp_angular = 2.0  # Proportional gain for angular velocity
        self.position_tolerance = 1  # meters
        
        # State variables
        self.current_waypoint = None
        self.arm_lowered = False
        self.arm_raised = False
        self.object_picked = False
        self.object_released = False
        self.state_start_time = None
        self.wait_time = 2.0  # seconds to wait after each action
        
        # Start state machine
        self.state = 'NAVIGATE_TO_PICKUP'
        self.state_start_time = rospy.Time.now()
        
        rospy.loginfo("ControlPickPlace initialized, starting pick and place sequence")
    
    def get_robot_pose(self):
        """
        Get robot pose in map frame using TF.
        
        Returns:
            tuple: (x, y, yaw) or (None, None, None) if transform unavailable
        """
        try:
            # Lookup transform from base_link to map
            transform = self.tf_buffer.lookup_transform(
                'map',  # target frame
                'lidar_link',  # source frame
                rospy.Time(0),  # latest available
                timeout=rospy.Duration(0.1)
            )
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract orientation (yaw)
            quat = transform.transform.rotation
            quat_array = [quat.x, quat.y, quat.z, quat.w]
            euler = tf.transformations.euler_from_quaternion(quat_array)
            yaw = euler[2]  # yaw is the third element
            
            return x, y, yaw
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1, f"TF lookup failed: {e}")
            return None, None, None
    
    def navigate_to_waypoint(self, waypoint):
        """
        Navigate robot to a waypoint using proportional control.
        
        Args:
            waypoint: numpy array [x, y] target position
        
        Returns:
            bool: True if waypoint reached, False otherwise
        """
        x, y, yaw = self.get_robot_pose()
        
        if x is None:
            return False
        
        # Calculate position error
        pos_error = waypoint - np.array([x, y])
        distance = np.linalg.norm(pos_error)
        
        # Check if waypoint reached
        if distance < self.position_tolerance:
            # Stop robot
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return True
        
        # Calculate desired heading
        desired_yaw = np.arctan2(pos_error[1], pos_error[0])
        
        # Calculate angle error (normalize to [-pi, pi])
        angle_error = desired_yaw - yaw
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        if (angle_error > self.angle_tolerance):
            angular_vel = self.max_angular_vel
        elif (angle_error < -self.angle_tolerance):
            angular_vel = -self.max_angular_vel
        else:
            angular_vel = 0.0
            linear_vel = self.max_linear_vel
        # Calculate velocities using proportional control
        # linear_vel = min(self.kp_linear * distance, self.max_linear_vel)
        
        # angular_vel = np.clip(self.kp_angular * angle_error, -self.max_angular_vel, self.max_angular_vel)
        
        # If angle error is large, rotate in place first
        if abs(angle_error) > self.angle_tolerance:
            linear_vel = 0.0
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        return False
    
    def _state_idle(self):
        """Idle state - do nothing"""
        pass
    
    def _state_navigate_to_pickup(self):
        """Navigate to pickup location"""
        if self.current_waypoint is None:
            self.current_waypoint = self.pickup_waypoint
            rospy.loginfo(f"Navigating to pickup location: ({self.pickup_waypoint[0]}, {self.pickup_waypoint[1]})")
        
        if self.navigate_to_waypoint(self.current_waypoint):
            rospy.loginfo("Reached pickup location")
            self.current_waypoint = None
            self.state = 'LOWER_ARM_PICKUP'
            self.state_start_time = rospy.Time.now()
    
    def _state_lower_arm_pickup(self):
        """Lower arm at pickup location"""
        if not self.arm_lowered:
            # Send command to lower arm
            cmd = Float64()
            cmd.data = 0.0
            self.shoulder_lift_pub.publish(cmd)
            self.arm_lowered = True
            rospy.loginfo("Lowering arm at pickup location")
            self.state_start_time = rospy.Time.now()
        
        # Wait for arm to lower
        if (rospy.Time.now() - self.state_start_time).to_sec() >= self.wait_time:
            rospy.loginfo("Arm lowered")
            self.state = 'PICK_OBJECT'
            self.state_start_time = rospy.Time.now()
            self.arm_lowered = False  # Reset for next use
    
    def pick(self):
        """Pick object with gripper"""
        if not self.object_picked:
            # Send pick command
            cmd = Bool()
            cmd.data = True
            self.gripper_pick_pub.publish(cmd)
            self.object_picked = True
            rospy.loginfo("Picking object")
            self.state_start_time = rospy.Time.now()
        
        # Wait for pick to complete
        if (rospy.Time.now() - self.state_start_time).to_sec() >= self.wait_time:
            rospy.loginfo("Object picked")
            self.state = 'RAISE_ARM_PICKUP'
            self.state_start_time = rospy.Time.now()
    
    def _state_raise_arm_pickup(self):
        """Raise arm after picking"""
        if not self.arm_raised:
            # Send command to raise arm
            cmd = Float64()
            cmd.data = -1.257
            self.shoulder_lift_pub.publish(cmd)
            self.arm_raised = True
            rospy.loginfo("Raising arm after pick")
            self.state_start_time = rospy.Time.now()
        
        # Wait for arm to raise
        if (rospy.Time.now() - self.state_start_time).to_sec() >= self.wait_time:
            rospy.loginfo("Arm raised")
            self.state = 'NAVIGATE_TO_BIN'
            self.state_start_time = rospy.Time.now()
            self.arm_raised = False  # Reset for next use
    
    def _state_navigate_to_bin(self):
        """Navigate to bin location"""
        if self.current_waypoint is None:
            self.current_waypoint = self.bin_waypoint
            rospy.loginfo(f"Navigating to bin location: ({self.bin_waypoint[0]}, {self.bin_waypoint[1]})")
        
        if self.navigate_to_waypoint(self.current_waypoint):
            rospy.loginfo("Reached bin location")
            self.current_waypoint = None
            self.state = 'LOWER_ARM_BIN'
            self.state_start_time = rospy.Time.now()
    
    def _state_lower_arm_bin(self):
        """Lower arm at bin location"""
        if not self.arm_lowered:
            # Send command to lower arm
            cmd = Float64()
            cmd.data = 0.0
            self.shoulder_lift_pub.publish(cmd)
            self.arm_lowered = True
            rospy.loginfo("Lowering arm at bin location")
            self.state_start_time = rospy.Time.now()
        
        # Wait for arm to lower
        if (rospy.Time.now() - self.state_start_time).to_sec() >= self.wait_time:
            rospy.loginfo("Arm lowered at bin")
            self.state = 'RELEASE_OBJECT'
            self.state_start_time = rospy.Time.now()
            self.arm_lowered = False  # Reset for next use
    
    def place_in_bin(self):
        """Release object from gripper"""
        if not self.object_released:
            # Send release command
            cmd = Bool()
            cmd.data = False
            self.gripper_pick_pub.publish(cmd)
            self.object_released = True
            rospy.loginfo("Releasing object")
            self.state_start_time = rospy.Time.now()
        
        # Wait for release to complete
        if (rospy.Time.now() - self.state_start_time).to_sec() >= self.wait_time:
            rospy.loginfo("Object released")
            self.state = 'DONE'
            self.state_start_time = rospy.Time.now()
    
    def _state_done(self):
        """Final state - operation complete"""
        # Stop robot
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        rospy.loginfo("Pick and place operation completed!")
        rospy.loginfo_throttle(5, "Operation complete. Robot stopped.")
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # Execute current state
            if self.state in self.states:
                self.states[self.state]()
            else:
                rospy.logwarn(f"Unknown state: {self.state}")
                self.state = 'IDLE'
            
            rate.sleep()


if __name__ == '__main__':
    try:
        controller = ControlPickPlace()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in control_pick_place: {e}")
        import traceback
        traceback.print_exc()
