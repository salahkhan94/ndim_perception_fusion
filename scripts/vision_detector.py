#!/usr/bin/env python3
"""
ROS node that subscribes to detections, RGB image, depth image, and camera_info,
and converts detected objects to point clouds.

This node:
- Subscribes to /camera/detections, /camera/rgb/image_raw, /camera/depth/image_raw, and /camera/rgb/camera_info
- Filters for blue-colored objects only
- Converts the first blue detected object to 3D point cloud using depth image
- Publishes point cloud as sensor_msgs/PointCloud2 in camera_link frame to /camera/depth/points
"""

import rospy
import numpy as np
import cv2
import tf2_ros
import tf.transformations
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2DArray, Detection2D


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
        self.detections = None
        self.rgb_received = False
        self.depth_received = False
        self.camera_info_received = False
        self.detections_received = False
        
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
        rospy.Subscriber('/camera/detections', Detection2DArray, self.detections_callback)
        
        # Publishers
        self.pointcloud_pub = rospy.Publisher('/camera/depth/points', PointCloud2, queue_size=1)
        self.pointcloud_lidar_pub = rospy.Publisher('/lidar/points', PointCloud2, queue_size=1)
        
        # TF2 buffer and listener for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Rate for publishing (disabled - now triggered by detections)
        # self.publish_rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Depth to PointCloud node initialized")
        rospy.loginfo("Waiting for detections, camera_info and TF transforms...")
        rospy.loginfo("Will only process blue-colored objects")
    
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
    
    def detections_callback(self, msg):
        """Callback for detections - process and generate point cloud for first blue object"""
        try:
            self.detections = msg
            self.detections_received = True
            
            # Process detections and generate point cloud
            self.process_detections_and_publish()
        except Exception as e:
            rospy.logwarn(f"Error in detections callback: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
    
    def is_blue_object(self, detection, rgb_array):
        """
        Check if an object is blue-colored by analyzing RGB values in the bounding box
        
        Args:
            detection: vision_msgs/Detection2D object
            rgb_array: numpy array of RGB values (height, width, 3)
        
        Returns:
            bool: True if object is blue, False otherwise
        """
        if rgb_array is None:
            return False
        
        try:
            # Get bounding box coordinates
            bbox = detection.bbox
            center_x = int(bbox.center.x)
            center_y = int(bbox.center.y)
            width = int(bbox.size_x)
            height = int(bbox.size_y)
            
            # Calculate bounding box bounds
            x_min = max(0, center_x - width // 2)
            x_max = min(rgb_array.shape[1], center_x + width // 2)
            y_min = max(0, center_y - height // 2)
            y_max = min(rgb_array.shape[0], center_y + height // 2)
            
            # Extract RGB values in the bounding box region
            bbox_region = rgb_array[y_min:y_max, x_min:x_max, :]
            
            if bbox_region.size == 0:
                return False
            
            # Calculate mean RGB values
            mean_r = np.mean(bbox_region[:, :, 0])
            mean_g = np.mean(bbox_region[:, :, 1])
            mean_b = np.mean(bbox_region[:, :, 2])
            
            # Check if blue is dominant (blue > red and blue > green)
            # Also check if blue is above a threshold (e.g., > 100)
            is_blue = (mean_b > mean_r) and (mean_b > mean_g) and (mean_b > 220)
            
            # rospy.loginfo(f"Object RGB: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}, is_blue={is_blue}")
            
            return is_blue
            
        except Exception as e:
            rospy.logwarn(f"Error checking if object is blue: {e}")
            return False
    
    def depth_to_pointcloud_bbox(self, depth_array, bbox, rgb_array=None):
        """
        Convert depth image to point cloud for a specific bounding box region
        
        Args:
            depth_array: numpy array of depth values (height, width) in meters
            bbox: vision_msgs/BoundingBox2D object
            rgb_array: optional numpy array of RGB values (height, width, 3)
        
        Returns:
            points: Nx3 or Nx6 numpy array (x, y, z) or (x, y, z, r, g, b)
        """
        if self.fx is None or self.fy is None:
            return None
        
        try:
            # Get bounding box coordinates
            center_x = int(bbox.center.x)
            center_y = int(bbox.center.y)
            width = int(bbox.size_x)
            height = int(bbox.size_y)
            
            # Calculate bounding box bounds
            x_min = max(0, center_x - width // 2)
            x_max = min(depth_array.shape[1], center_x + width // 2)
            y_min = max(0, center_y - height // 2)
            y_max = min(depth_array.shape[0], center_y + height // 2)
            
            # Extract depth region for bounding box
            depth_region = depth_array[y_min:y_max, x_min:x_max]
            
            if depth_region.size == 0:
                return None
            
            # Create coordinate grids for the bounding box region
            u_local, v_local = np.meshgrid(
                np.arange(x_min, x_max),
                np.arange(y_min, y_max)
            )
            
            # Convert to 3D points using camera intrinsics
            x = (u_local - self.cx) * depth_region / self.fx
            y = (v_local - self.cy) * depth_region / self.fy
            z = depth_region
            
            # Stack into point cloud
            points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
            
            # Filter out invalid points (zero, NaN, or Inf depth)
            valid_mask = (z.flatten() > 0) & ~np.isnan(z.flatten()) & ~np.isinf(z.flatten())
            points_3d = points_3d[valid_mask]
            
            # Add RGB colors if available
            if rgb_array is not None:
                rgb_region = rgb_array[y_min:y_max, x_min:x_max, :]
                rgb_flat = rgb_region.reshape(-1, 3)[valid_mask]
                # Normalize RGB to 0-1 range (PointCloud2 expects float)
                rgb_normalized = rgb_flat.astype(np.float32) / 255.0
                points = np.hstack([points_3d, rgb_normalized])
            else:
                points = points_3d
            
            return points
            
        except Exception as e:
            rospy.logwarn(f"Error converting bbox to point cloud: {e}")
            return None
    
    def depth_to_pointcloud(self, depth_array, rgb_array=None):
        """
        Convert depth image to point cloud (full image - currently disabled)
        
        Args:
            depth_array: numpy array of depth values (height, width) in meters
            rgb_array: optional numpy array of RGB values (height, width, 3)
        
        Returns:
            points: Nx3 or Nx6 numpy array (x, y, z) or (x, y, z, r, g, b)
        """
        # This method is kept for compatibility but is not used anymore
        # Point cloud generation is now done per-detection via depth_to_pointcloud_bbox
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
    
    def transform_points_to_lidar_frame(self, points, timestamp):
        """
        Transform points from camera_link frame to lidar_link frame using TF2
        
        Args:
            points: Nx3 or Nx6 numpy array (x, y, z) or (x, y, z, r, g, b) in camera_link frame
            timestamp: rospy.Time for the transform lookup
        
        Returns:
            transformed_points: Nx3 or Nx6 numpy array in lidar_link frame, or None if transform fails
        """
        try:
            # Lookup transform from camera_link to lidar_link
            transform = self.tf_buffer.lookup_transform(
                'lidar_link',  # target frame
                'camera_link',  # source frame
                timestamp,
                timeout=rospy.Duration(0.1)
            )
            
            # Extract translation and rotation from transform
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            rotation_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            
            # Convert quaternion to rotation matrix
            rotation_matrix = tf.transformations.quaternion_matrix(rotation_quat)[:3, :3]
            
            # Extract XYZ coordinates (first 3 columns)
            xyz = points[:, :3]
            
            # Apply transform: R * p + t
            # Transform points: new_point = R @ point + translation
            xyz_transformed = (rotation_matrix @ xyz.T).T + translation
            
            # If points have RGB, preserve them
            if points.shape[1] == 6:
                rgb = points[:, 3:6]
                transformed_points = np.hstack([xyz_transformed, rgb])
            else:
                transformed_points = xyz_transformed
            
            return transformed_points
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF transform lookup failed: {e}")
            return None
        except Exception as e:
            rospy.logwarn(f"Error transforming points: {e}")
            return None
    
    def process_detections_and_publish(self):
        """
        Process detections, filter for blue objects only, and generate point cloud for first blue object
        """
        if not self.camera_info_received:
            return
        
        if not self.depth_received:
            return
        
        if not self.detections_received or self.detections is None:
            return
        
        try:
            # Convert depth image to numpy array
            depth_array = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="32FC1")
            
            # Get RGB image if available
            rgb_array = None
            if self.rgb_received and self.rgb_image is not None:
                try:
                    rgb_array = self.bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding="rgb8")
                except:
                    pass  # Continue without RGB if conversion fails
            
            # Process detections
            detections_list = self.detections.detections
            
            if len(detections_list) == 0:
                rospy.logdebug("No detections received")
                return
            
            # Filter for blue objects only and find first blue object
            selected_detection = None
            for detection in detections_list:
                # Check if object is blue
                if rgb_array is not None and self.is_blue_object(detection, rgb_array):
                    # Found first blue object
                    selected_detection = detection
                    rospy.loginfo_throttle(2, f"Selected blue object for point cloud generation")
                    break
                else:
                    rospy.logdebug("Skipping non-blue object")
            
            # If no blue object found, return
            if selected_detection is None:
                rospy.logwarn_throttle(2, "No blue objects found in detections")
                return
            
            # Generate point cloud for the selected object's bounding box
            points = self.depth_to_pointcloud_bbox(
                depth_array,
                selected_detection.bbox,
                rgb_array
            )
            
            if points is None or len(points) == 0:
                rospy.logwarn_throttle(2, "No valid points generated for selected object")
                return
            
            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_link"  # Point cloud in camera frame
            
            # Create and publish PointCloud2 message in camera_link frame
            pointcloud_msg = self.create_pointcloud2_msg(points, header)
            self.pointcloud_pub.publish(pointcloud_msg)
            
            # Transform point cloud to lidar_link frame and publish
            # points_lidar_frame = self.transform_points_to_lidar_frame(points, header.stamp)
            # if points_lidar_frame is not None:
            #     # Create header for lidar frame
            #     header_lidar = Header()
            #     header_lidar.stamp = header.stamp
            #     header_lidar.frame_id = "lidar_link"
                
            #     # Create and publish PointCloud2 message in lidar_link frame
            #     pointcloud_lidar_msg = self.create_pointcloud2_msg(points_lidar_frame, header_lidar)
            #     self.pointcloud_lidar_pub.publish(pointcloud_lidar_msg)
            
            rospy.loginfo_throttle(2, f"Published point cloud with {len(points)} points for blue object")
            
        except Exception as e:
            rospy.logwarn(f"Error processing detections: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
    
    def process_and_publish(self):
        """
        Process depth image and publish point cloud (DISABLED - now using process_detections_and_publish)
        """
        # This method is disabled - point cloud generation is now triggered by detections
        pass
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Starting depth to pointcloud conversion (detection-based)...")
        rospy.loginfo("Point cloud generation is now triggered by detections")
        
        # Spin and wait for callbacks
        rospy.spin()


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

