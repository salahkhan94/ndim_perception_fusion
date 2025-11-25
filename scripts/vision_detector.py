#!/usr/bin/env python3
"""
ROS node that subscribes to RGB image, depth image, and camera_info,
and converts the depth image to a point cloud in the camera_link frame.

This node:
- Subscribes to /camera/rgb/image_raw, /camera/depth/image_raw, and /camera/rgb/camera_info
- Converts depth image to 3D point cloud using camera intrinsics
- Publishes point cloud as sensor_msgs/PointCloud2
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from std_msgs.msg import Header


class DepthToPointCloud:
    """Convert depth images to point clouds using camera intrinsics"""
    
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Store latest messages
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.rgb_received = False
        self.depth_received = False
        self.camera_info_received = False
        
        # Camera intrinsics (will be updated from camera_info)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_width = None
        self.image_height = None
        
        # Subscribers
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)
        
        # Publisher
        self.pointcloud_pub = rospy.Publisher('/camera/depth/points', PointCloud2, queue_size=1)
        
        # Rate for publishing
        self.publish_rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Depth to PointCloud node initialized")
        rospy.loginfo("Waiting for camera_info to start publishing point clouds...")
    
    def rgb_callback(self, msg):
        """Callback for RGB image"""
        try:
            self.rgb_image = msg
            self.rgb_received = True
        except Exception as e:
            rospy.logwarn(f"Error in RGB callback: {e}")
    
    def depth_callback(self, msg):
        """Callback for depth image"""
        try:
            self.depth_image = msg
            self.depth_received = True
        except Exception as e:
            rospy.logwarn(f"Error in depth callback: {e}")
    
    def camera_info_callback(self, msg):
        """Callback for camera_info - extract intrinsics"""
        try:
            self.camera_info = msg
            
            # Extract camera intrinsics
            if len(msg.K) >= 9:
                # K is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                self.fx = msg.K[0]
                self.fy = msg.K[4]
                self.cx = msg.K[2]
                self.cy = msg.K[5]
                self.image_width = msg.width
                self.image_height = msg.height
                self.camera_info_received = True
                rospy.loginfo_once(f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                                 f"cx={self.cx:.2f}, cy={self.cy:.2f}, size={self.image_width}x{self.image_height}")
        except Exception as e:
            rospy.logwarn(f"Error in camera_info callback: {e}")
    
    def depth_to_pointcloud(self, depth_array, rgb_array=None):
        """
        Convert depth image to point cloud
        
        Args:
            depth_array: numpy array of depth values (height, width) in meters
            rgb_array: optional numpy array of RGB values (height, width, 3)
        
        Returns:
            points: Nx3 or Nx6 numpy array (x, y, z) or (x, y, z, r, g, b)
        """
        if self.fx is None or self.fy is None:
            return None
        
        height, width = depth_array.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D points using camera intrinsics
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # z = depth
        x = (u - self.cx) * depth_array / self.fx
        y = (v - self.cy) * depth_array / self.fy
        z = depth_array
        
        # Stack into point cloud
        points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Filter out invalid points (zero, NaN, or Inf depth)
        valid_mask = (z.flatten() > 0) & ~np.isnan(z.flatten()) & ~np.isinf(z.flatten())
        points_3d = points_3d[valid_mask]
        
        # Add RGB colors if available
        if rgb_array is not None:
            rgb_flat = rgb_array.reshape(-1, 3)[valid_mask]
            # Normalize RGB to 0-1 range (PointCloud2 expects float)
            rgb_normalized = rgb_flat.astype(np.float32) / 255.0
            points = np.hstack([points_3d, rgb_normalized])
        else:
            points = points_3d
        
        return points
    
    def create_pointcloud2_msg(self, points, header):
        """
        Create sensor_msgs/PointCloud2 message from point array
        
        Args:
            points: Nx3 or Nx6 numpy array (x, y, z) or (x, y, z, r, g, b)
            header: std_msgs/Header with frame_id and timestamp
        
        Returns:
            pointcloud_msg: sensor_msgs/PointCloud2 message
        """
        msg = PointCloud2()
        msg.header = header
        
        if points is None or len(points) == 0:
            # Return empty point cloud
            msg.width = 0
            msg.height = 1
            msg.is_dense = True
            return msg
        
        num_points = len(points)
        
        # Determine if we have RGB colors
        has_rgb = points.shape[1] == 6
        
        # Define point fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        if has_rgb:
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
            point_step = 16  # 4 bytes per float * 3 + 4 bytes for RGB
        else:
            point_step = 12  # 4 bytes per float * 3
        
        # Set message properties
        msg.width = num_points
        msg.height = 1
        msg.is_dense = False  # May have invalid points
        msg.point_step = point_step
        msg.row_step = point_step * num_points
        msg.fields = fields
        
        # Convert points to byte array
        if has_rgb:
            # Pack RGB into single uint32: RGB -> 0x00RRGGBB
            xyz = points[:, :3].astype(np.float32)
            rgb = points[:, 3:6].astype(np.float32)
            
            # Convert RGB to uint32 format (0x00RRGGBB)
            r = (rgb[:, 0] * 255).astype(np.uint32)
            g = (rgb[:, 1] * 255).astype(np.uint32)
            b = (rgb[:, 2] * 255).astype(np.uint32)
            rgb_packed = (r << 16) | (g << 8) | b
            
            # Create structured array
            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.uint32)
            ])
            structured_array = np.empty(num_points, dtype=dtype)
            structured_array['x'] = xyz[:, 0]
            structured_array['y'] = xyz[:, 1]
            structured_array['z'] = xyz[:, 2]
            structured_array['rgb'] = rgb_packed
            
            msg.data = structured_array.tobytes()
        else:
            # Just XYZ
            xyz = points.astype(np.float32)
            msg.data = xyz.tobytes()
        
        return msg
    
    def process_and_publish(self):
        """Process depth image and publish point cloud"""
        if not self.camera_info_received:
            return
        
        if not self.depth_received:
            return
        
        try:
            # Convert depth image to numpy array
            depth_array = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="32FC1")
            
            # Debug: Log depth statistics
            valid_depths = depth_array[depth_array > 0]
            if len(valid_depths) > 0:
                rospy.loginfo_throttle(5, f"Received depth image stats: min={valid_depths.min():.3f}m, "
                                         f"max={valid_depths.max():.3f}m, "
                                         f"mean={valid_depths.mean():.3f}m, "
                                         f"std={valid_depths.std():.3f}m, "
                                         f"valid_pixels={len(valid_depths)}/{depth_array.size}")
            else:
                rospy.logwarn_throttle(5, "No valid depth values found in depth image!")
            
            # Get RGB image if available
            rgb_array = None
            if self.rgb_received and self.rgb_image is not None:
                try:
                    rgb_array = self.bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding="rgb8")
                except:
                    pass  # Continue without RGB if conversion fails
            
            # Convert depth to point cloud
            points = self.depth_to_pointcloud(depth_array, rgb_array)
            
            # Debug: Log point cloud statistics
            if points is not None and len(points) > 0:
                z_values = points[:, 2]  # Z coordinates
                rospy.loginfo_throttle(5, f"Point cloud Z stats: min={z_values.min():.3f}m, "
                                         f"max={z_values.max():.3f}m, "
                                         f"mean={z_values.mean():.3f}m, "
                                         f"std={z_values.std():.3f}m")
            
            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_link"  # Point cloud in camera frame
            
            # Create and publish PointCloud2 message
            pointcloud_msg = self.create_pointcloud2_msg(points, header)
            self.pointcloud_pub.publish(pointcloud_msg)
            
            if points is not None:
                rospy.loginfo_throttle(2, f"Published point cloud with {len(points)} points")
            
        except Exception as e:
            rospy.logwarn(f"Error processing depth image: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Starting depth to pointcloud conversion...")
        
        while not rospy.is_shutdown():
            self.process_and_publish()
            self.publish_rate.sleep()


if __name__ == '__main__':
    try:
        converter = DepthToPointCloud()
        converter.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in depth_to_pointcloud node: {e}")
        import traceback
        traceback.print_exc()

