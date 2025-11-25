#!/usr/bin/env python3
"""
Example script showing how to extract depth/Z values from the depth image
published by the PyBullet simulation node.

The depth image is published as sensor_msgs/Image with encoding "32FC1",
where each pixel value represents the depth in meters from the camera.
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DepthExtractor:
    """Example class for extracting depth values from depth image"""
    
    def __init__(self):
        rospy.init_node('depth_extractor', anonymous=True)
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Subscribe to depth image topic
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        # Store latest depth image
        self.depth_array = None
        self.depth_image_received = False
        
        rospy.loginfo("Depth extractor initialized. Waiting for depth images...")
    
    def depth_callback(self, msg):
        """Callback function when a new depth image is received"""
        try:
            # Convert ROS Image message to numpy array
            # encoding is "32FC1" (32-bit float, single channel)
            depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            
            # depth_array is now a numpy array of shape (height, width)
            # Each value is the depth in meters from the camera
            self.depth_array = depth_array
            self.depth_image_received = True
            
            rospy.loginfo_once("Depth image received successfully")
            
        except Exception as e:
            rospy.logwarn(f"Error converting depth image: {e}")
    
    def get_depth_at_pixel(self, u, v):
        """
        Get depth value at a specific pixel coordinate
        
        Args:
            u: x coordinate (column) in image
            v: y coordinate (row) in image
        
        Returns:
            depth: Depth value in meters, or None if invalid
        """
        if self.depth_array is None:
            rospy.logwarn("No depth image available")
            return None
        
        # Check bounds
        height, width = self.depth_array.shape
        if u < 0 or u >= width or v < 0 or v >= height:
            rospy.logwarn(f"Pixel coordinates ({u}, {v}) out of bounds. Image size: {width}x{height}")
            return None
        
        depth = self.depth_array[v, u]
        
        # Check for invalid depth values (NaN, Inf, or 0)
        if np.isnan(depth) or np.isinf(depth) or depth <= 0:
            return None
        
        return depth
    
    def get_depth_in_region(self, u_min, u_max, v_min, v_max):
        """
        Get depth values in a rectangular region
        
        Args:
            u_min, u_max: x coordinate range (columns)
            v_min, v_max: y coordinate range (rows)
        
        Returns:
            depth_region: Numpy array of depth values in the region
        """
        if self.depth_array is None:
            rospy.logwarn("No depth image available")
            return None
        
        # Clip to image bounds
        height, width = self.depth_array.shape
        u_min = max(0, min(u_min, width - 1))
        u_max = max(0, min(u_max, width - 1))
        v_min = max(0, min(v_min, height - 1))
        v_max = max(0, min(v_max, height - 1))
        
        return self.depth_array[v_min:v_max+1, u_min:u_max+1]
    
    def get_min_depth_in_region(self, u_min, u_max, v_min, v_max):
        """Get minimum depth value in a region (closest object)"""
        depth_region = self.get_depth_in_region(u_min, u_max, v_min, v_max)
        if depth_region is None:
            return None
        
        # Filter out invalid values
        valid_depths = depth_region[(depth_region > 0) & ~np.isnan(depth_region) & ~np.isinf(depth_region)]
        
        if len(valid_depths) == 0:
            return None
        
        return np.min(valid_depths)
    
    def get_max_depth_in_region(self, u_min, u_max, v_min, v_max):
        """Get maximum depth value in a region (farthest object)"""
        depth_region = self.get_depth_in_region(u_min, u_max, v_min, v_max)
        if depth_region is None:
            return None
        
        # Filter out invalid values
        valid_depths = depth_region[(depth_region > 0) & ~np.isnan(depth_region) & ~np.isinf(depth_region)]
        
        if len(valid_depths) == 0:
            return None
        
        return np.max(valid_depths)
    
    def get_mean_depth_in_region(self, u_min, u_max, v_min, v_max):
        """Get mean depth value in a region"""
        depth_region = self.get_depth_in_region(u_min, u_max, v_min, v_max)
        if depth_region is None:
            return None
        
        # Filter out invalid values
        valid_depths = depth_region[(depth_region > 0) & ~np.isnan(depth_region) & ~np.isinf(depth_region)]
        
        if len(valid_depths) == 0:
            return None
        
        return np.mean(valid_depths)
    
    def get_full_depth_array(self):
        """
        Get the full depth array
        
        Returns:
            depth_array: Numpy array of shape (height, width) with depth values in meters
        """
        return self.depth_array
    
    def depth_to_point_cloud(self, camera_intrinsics=None):
        """
        Convert depth image to 3D point cloud
        
        Args:
            camera_intrinsics: 3x3 camera intrinsic matrix
                             [[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]]
                             If None, uses default values based on image size
        
        Returns:
            points: Nx3 numpy array of 3D points (X, Y, Z) in camera frame
        """
        if self.depth_array is None:
            rospy.logwarn("No depth image available")
            return None
        
        height, width = self.depth_array.shape
        
        # Default camera intrinsics (if not provided)
        if camera_intrinsics is None:
            # Assuming FOV of 60 degrees and image size
            fov = 60.0  # degrees
            fx = fy = width / (2.0 * np.tan(np.radians(fov) / 2.0))
            cx = width / 2.0
            cy = height / 2.0
            camera_intrinsics = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])
        
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D points
        z = self.depth_array
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Filter out invalid points (zero or invalid depth)
        valid_mask = (z.flatten() > 0) & ~np.isnan(z.flatten()) & ~np.isinf(z.flatten())
        points = points[valid_mask]
        
        return points


def example_usage():
    """Example of how to use the DepthExtractor"""
    extractor = DepthExtractor()
    
    # Wait for first depth image
    rospy.sleep(1.0)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        if extractor.depth_image_received:
            # Example 1: Get depth at center pixel
            center_u = 320  # Assuming 640x480 image
            center_v = 240
            depth = extractor.get_depth_at_pixel(center_u, center_v)
            if depth is not None:
                rospy.loginfo(f"Depth at center pixel ({center_u}, {center_v}): {depth:.3f} m")
            
            # Example 2: Get depth statistics in a region
            u_min, u_max = 200, 400
            v_min, v_max = 150, 300
            min_depth = extractor.get_min_depth_in_region(u_min, u_max, v_min, v_max)
            mean_depth = extractor.get_mean_depth_in_region(u_min, u_max, v_min, v_max)
            max_depth = extractor.get_max_depth_in_region(u_min, u_max, v_min, v_max)
            
            if min_depth is not None:
                rospy.loginfo(f"Depth in region [{u_min}:{u_max}, {v_min}:{v_max}]: "
                            f"min={min_depth:.3f}m, mean={mean_depth:.3f}m, max={max_depth:.3f}m")
            
            # Example 3: Get full depth array
            depth_array = extractor.get_full_depth_array()
            if depth_array is not None:
                rospy.loginfo(f"Full depth array shape: {depth_array.shape}")
                rospy.loginfo(f"Depth range: [{np.nanmin(depth_array):.3f}, {np.nanmax(depth_array):.3f}] m")
            
            # Example 4: Convert to point cloud
            point_cloud = extractor.depth_to_point_cloud()
            if point_cloud is not None:
                rospy.loginfo(f"Point cloud has {len(point_cloud)} valid points")
        
        rate.sleep()


if __name__ == '__main__':
    try:
        example_usage()
    except rospy.ROSInterruptException:
        pass

