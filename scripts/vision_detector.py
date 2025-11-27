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
from geometry_msgs.msg import TransformStamped, PoseStamped, Pose, Point, Quaternion
from vision_msgs.msg import Detection2DArray, Detection2D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    rospy.logwarn("Open3D not available. Cuboid pose estimation will be disabled.")


class CuboidPoseEstimator:
    """
    Estimates the 6DOF pose of a cuboid from a point cloud using RANSAC plane fitting.
    Handles both 1-face and 2-face visible scenarios.
    """
    
    def __init__(self, dimensions=None, tf_buffer=None):
        """
        Initialize the cuboid pose estimator
        
        Args:
            dimensions: List of cuboid dimensions [L, W, H] (sorted smallest to largest)
            tf_buffer: TF2 buffer for coordinate frame transformations (optional)
        """
        # Default dimensions from simple_box.urdf: 0.3 x 0.6 x 0.4
        # Sorted: [0.3, 0.4, 0.6]
        self.dimensions = dimensions if dimensions is not None else [0.3, 0.4, 0.6]
        self.dimensions = sorted(self.dimensions)  # Ensure sorted
        
        # TF buffer for transformations
        self.tf_buffer = tf_buffer
        
        # RANSAC parameters
        self.ransac_distance_threshold = 0.01  # 1cm threshold (adjust based on point cloud density)
        self.ransac_n_points = 3
        self.ransac_n_iterations = 1000
        
        # Thresholds for plane detection
        self.min_plane_inliers_ratio = 0.2  # At least 20% of points for a valid plane
        self.min_plane_inliers_count = 50   # Minimum absolute count
        self.orthogonality_tolerance = np.deg2rad(10)  # 10 degrees tolerance for perpendicular planes
        
        # Edge detection threshold (distance from both planes)
        self.edge_distance_threshold = 0.02  # 2cm
        
        rospy.loginfo(f"CuboidPoseEstimator initialized with dimensions: {self.dimensions}")
    
    def estimate_pose(self, points, timestamp=None):
        """
        Main method to estimate cuboid pose from point cloud
        
        Args:
            points: Nx3 numpy array of 3D points (x, y, z) in camera_link frame
            timestamp: rospy.Time for TF lookups (optional)
        
        Returns:
            pose_dict: Dictionary with 'position', 'orientation', 'frame_id', 'success'
        """
        if not OPEN3D_AVAILABLE:
            rospy.logwarn("Open3D not available, cannot estimate pose")
            return None
        
        if points is None or len(points) < 10:
            rospy.logwarn("Insufficient points for pose estimation")
            return None
        
        try:
            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            # Fit planes using RANSAC
            planes = self._fit_planes_ransac(pcd)
            
            if len(planes) == 0:
                rospy.logwarn("No planes detected")
                return None
            
            # Determine if 1-face or 2-face scenario
            face_count = self._detect_face_count(planes, len(points))
            
            if face_count == 1:
                rospy.loginfo("Detected 1-face scenario")
                return self._estimate_pose_1face(planes[0], points, timestamp)
            elif face_count == 2:
                rospy.loginfo("Detected 2-face scenario")
                return self._estimate_pose_2face(planes[0], planes[1], points)
            else:
                rospy.logwarn(f"Unexpected number of faces: {face_count}")
                return None
                
        except Exception as e:
            rospy.logwarn(f"Error in pose estimation: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
            return None
    
    def _fit_planes_ransac(self, pcd):
        """
        Fit planes to point cloud using RANSAC
        
        Args:
            pcd: Open3D PointCloud
        
        Returns:
            List of plane dictionaries: [{'normal': array, 'point': array, 'inliers': indices}, ...]
        """
        planes = []
        remaining_pcd = pcd
        
        # Fit first plane
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=self.ransac_distance_threshold,
            ransac_n=self.ransac_n_points,
            num_iterations=self.ransac_n_iterations
        )
        
        if len(inliers) < self.min_plane_inliers_count:
            return planes
        
        # Extract plane parameters: [a, b, c, d] where ax + by + cz + d = 0
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Get a point on the plane (project origin onto plane)
        plane_point = -d * normal / (np.dot(normal, normal))
        
        planes.append({
            'normal': normal,
            'point': plane_point,
            'inliers': inliers,
            'inlier_count': len(inliers)
        })
        
        # Remove inliers and try to fit second plane
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        
        if len(remaining_pcd.points) < self.min_plane_inliers_count:
            return planes
        
        # Fit second plane
        plane_model2, inliers2 = remaining_pcd.segment_plane(
            distance_threshold=self.ransac_distance_threshold,
            ransac_n=self.ransac_n_points,
            num_iterations=self.ransac_n_iterations
        )
        
        if len(inliers2) >= self.min_plane_inliers_count:
            a2, b2, c2, d2 = plane_model2
            normal2 = np.array([a2, b2, c2])
            normal2 = normal2 / np.linalg.norm(normal2)
            plane_point2 = -d2 * normal2 / (np.dot(normal2, normal2))
            
            planes.append({
                'normal': normal2,
                'point': plane_point2,
                'inliers': inliers2,
                'inlier_count': len(inliers2)
            })
        
        return planes
    
    def _detect_face_count(self, planes, total_points):
        """
        Determine if we have 1-face or 2-face scenario
        
        Args:
            planes: List of detected planes
            total_points: Total number of points in point cloud
        
        Returns:
            int: 1 or 2
        """
        if len(planes) == 0:
            return 0
        
        if len(planes) == 1:
            return 1
        
        # Check if second plane has enough inliers
        plane2_ratio = planes[1]['inlier_count'] / total_points
        
        if plane2_ratio >= self.min_plane_inliers_ratio:
            return 2
        else:
            return 1
    
    def _estimate_pose_1face(self, plane, points, timestamp=None):
        """
        Estimate pose when only 1 face is visible.
        Transforms points to lidar_link, does PCA, and ensures only yaw rotation.
        
        Args:
            plane: Plane dictionary with 'normal', 'point', 'inliers'
            points: Nx3 numpy array of all points in camera_link frame
            timestamp: rospy.Time for TF lookup (optional)
        
        Returns:
            pose_dict: Dictionary with pose information in camera_link frame
        """
        # Get inlier points
        inlier_points = points[plane['inliers']]
        plane_normal_camera = plane['normal']
        
        if len(inlier_points) < 3:
            rospy.logwarn("Not enough inlier points for pose estimation")
            return None
        
        # Transform points to lidar_link frame
        if self.tf_buffer is not None and timestamp is not None:
            try:
                # Lookup transform from camera_link to lidar_link
                transform = self.tf_buffer.lookup_transform(
                    'lidar_link',
                    'camera_link',
                    timestamp,
                    timeout=rospy.Duration(0.1)
                )
                
                # Extract transform
                t = transform.transform.translation
                r = transform.transform.rotation
                translation_tf = np.array([t.x, t.y, t.z])
                rotation_quat = np.array([r.x, r.y, r.z, r.w])
                rotation_matrix_tf = tf.transformations.quaternion_matrix(rotation_quat)[:3, :3]
                
                # Transform points: R * p + t
                inlier_points_lidar = (rotation_matrix_tf @ inlier_points.T).T + translation_tf
                
                # Transform plane normal: R * n (rotation only, no translation)
                plane_normal_lidar = rotation_matrix_tf @ plane_normal_camera
                plane_normal_lidar = plane_normal_lidar / np.linalg.norm(plane_normal_lidar)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed, using camera_link frame: {e}")
                inlier_points_lidar = inlier_points
                plane_normal_lidar = plane_normal_camera
        else:
            rospy.logwarn("TF buffer or timestamp not available, using camera_link frame")
            inlier_points_lidar = inlier_points
            plane_normal_lidar = plane_normal_camera
        
        # Center the points
        center_surface_lidar = np.mean(inlier_points_lidar, axis=0)
        centered_points_lidar = inlier_points_lidar - center_surface_lidar
        
        # Do PCA on transformed points
        if len(centered_points_lidar) < 3:
            rospy.logwarn("Not enough points for PCA")
            return None
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points_lidar.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Print eigenvalues and eigenvectors for debugging
        rospy.loginfo("=" * 60)
        rospy.loginfo("PCA Results (in lidar_link frame):")
        rospy.loginfo(f"Eigenvalues: {eigenvalues}")
        rospy.loginfo(f"Eigenvector 1 (largest): {eigenvectors[:, 0]}")
        rospy.loginfo(f"Eigenvector 2 (second): {eigenvectors[:, 1]}")
        rospy.loginfo(f"Eigenvector 3 (smallest): {eigenvectors[:, 2]}")
        rospy.loginfo("=" * 60)
        
        # u1: horizontal eigenvector (largest eigenvalue) - use for yaw
        u1 = eigenvectors[:, 0]
        u1 = u1 / np.linalg.norm(u1)
        
        # u2: vertical eigenvector (second largest eigenvalue) - should be vertical
        u2 = eigenvectors[:, 1]
        u2 = u2 / np.linalg.norm(u2)
        
        # Project u1 onto XY plane for yaw calculation
        u1_horizontal = np.array([u1[0], u1[1], 0.0])
        u1_horizontal = u1_horizontal / np.linalg.norm(u1_horizontal) if np.linalg.norm(u1_horizontal) > 1e-6 else np.array([1.0, 0.0, 0.0])
        
        # Calculate yaw from horizontal eigenvector
        yaw = np.arctan2(u1_horizontal[1], u1_horizontal[0])
        
        # Ensure u2 points up (vertical)
        if u2[2] < 0:
            u2 = -u2
        
        # Measure spans to identify dimensions
        # Project points onto u1 (horizontal) and u2 (vertical)
        proj_u1 = np.dot(centered_points_lidar, u1)
        proj_u2 = np.dot(centered_points_lidar, u2)
        
        horizontal_span = np.max(proj_u1) - np.min(proj_u1)
        vertical_span = np.max(proj_u2) - np.min(proj_u2)
        
        rospy.loginfo(f"Horizontal span (u1): {horizontal_span:.4f} m")
        rospy.loginfo(f"Vertical span (u2): {vertical_span:.4f} m")
        rospy.loginfo(f"Cuboid dimensions: {self.dimensions}")
        
        # Match spans to dimensions to identify which face we're seeing
        # The remaining dimension is the depth
        horizontal_dim_idx = None
        vertical_dim_idx = None
        
        # Find closest matching dimensions
        horizontal_error = [abs(horizontal_span - dim) for dim in self.dimensions]
        vertical_error = [abs(vertical_span - dim) for dim in self.dimensions]
        
        horizontal_dim_idx = np.argmin(horizontal_error)
        vertical_dim_idx = np.argmin(vertical_error)
        
        # If same dimension matched, pick the next best for one of them
        if horizontal_dim_idx == vertical_dim_idx:
            # Find second best match for vertical
            vertical_error_copy = vertical_error.copy()
            vertical_error_copy[vertical_dim_idx] = float('inf')
            vertical_dim_idx = np.argmin(vertical_error_copy)
        
        horizontal_dim = self.dimensions[horizontal_dim_idx]
        vertical_dim = self.dimensions[vertical_dim_idx]
        
        # Depth is the remaining dimension
        depth_dim_idx = [i for i in range(3) if i != horizontal_dim_idx and i != vertical_dim_idx][0]
        depth_dim = self.dimensions[depth_dim_idx]
        
        rospy.loginfo(f"Identified: horizontal={horizontal_dim:.4f}m (idx={horizontal_dim_idx}), "
                     f"vertical={vertical_dim:.4f}m (idx={vertical_dim_idx}), "
                     f"depth={depth_dim:.4f}m (idx={depth_dim_idx})")
        
        # Build rotation matrix using Euler angles: roll=0, pitch=0, yaw=calculated
        # Since cuboid is flat on ground, only yaw rotation is needed
        roll = 0.0
        pitch = 0.0
        # yaw is already calculated above
        
        # Convert Euler angles (roll, pitch, yaw) to rotation matrix
        # Using ZYX convention (yaw around Z, pitch around Y, roll around X)
        rotation_matrix_lidar = tf.transformations.euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        
        # Calculate translation in lidar_link: move inward along plane normal
        translation_lidar = center_surface_lidar - plane_normal_lidar * (depth_dim / 2.0)
        
        rospy.loginfo(f"Translation in lidar_link: {translation_lidar}")
        rospy.loginfo(f"Plane normal in lidar_link: {plane_normal_lidar}")
        rospy.loginfo(f"Yaw angle: {np.rad2deg(yaw):.2f} degrees")
        
        # Transform pose back to camera_link frame for return
        if self.tf_buffer is not None and timestamp is not None:
            try:
                # Inverse transform: camera_link = T^-1 * lidar_link
                transform = self.tf_buffer.lookup_transform(
                    'camera_link',
                    'lidar_link',
                    timestamp,
                    timeout=rospy.Duration(0.1)
                )
                
                t = transform.transform.translation
                r = transform.transform.rotation
                translation_tf_inv = np.array([t.x, t.y, t.z])
                rotation_quat_inv = np.array([r.x, r.y, r.z, r.w])
                rotation_matrix_tf_inv = tf.transformations.quaternion_matrix(rotation_quat_inv)[:3, :3]
                
                # Transform translation: R^-1 * (t - t_tf)
                translation_camera = rotation_matrix_tf_inv @ (translation_lidar - translation_tf_inv)
                
                # Transform rotation: R^-1 * R_lidar
                rotation_matrix_camera = rotation_matrix_tf_inv @ rotation_matrix_lidar
                
            except Exception as e:
                rospy.logwarn(f"Failed to transform pose back to camera_link: {e}")
                translation_camera = translation_lidar
                rotation_matrix_camera = rotation_matrix_lidar
        else:
            translation_camera = translation_lidar
            rotation_matrix_camera = rotation_matrix_lidar
        
        # Convert to quaternion
        quaternion = tf.transformations.quaternion_from_matrix(
            np.vstack([np.column_stack([rotation_matrix_camera, translation_camera]), [0, 0, 0, 1]])
        )
        
        return {
            'position': translation_camera,
            'orientation': quaternion,
            'rotation_matrix': rotation_matrix_camera,
            'success': True,
            'face_count': 1,
            'yaw': yaw,
            'horizontal_dim': horizontal_dim,
            'vertical_dim': vertical_dim,
            'depth_dim': depth_dim
        }
    
    def _estimate_pose_2face(self, plane1, plane2, points):
        """
        Estimate pose when 2 faces are visible (L-shape)
        
        Args:
            plane1: First plane dictionary
            plane2: Second plane dictionary
            points: Nx3 numpy array of all points
        
        Returns:
            pose_dict: Dictionary with pose information
        """
        normal1 = plane1['normal']
        normal2 = plane2['normal']
        
        # Validate normal directions
        inlier_points1 = points[plane1['inliers']]
        inlier_points2 = points[plane2['inliers']]
        normal1 = self._validate_normal_direction(normal1, inlier_points1)
        normal2 = self._validate_normal_direction(normal2, inlier_points2)
        
        # Validate orthogonality
        angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
        if abs(angle - np.pi/2) > self.orthogonality_tolerance:
            rospy.logwarn(f"Planes are not perpendicular: {np.rad2deg(angle):.1f} degrees")
            # Still proceed, but warn
        
        # Compute third axis (cross product)
        axis3 = np.cross(normal1, normal2)
        axis3 = axis3 / np.linalg.norm(axis3)
        
        # Ensure right-handed coordinate system
        # Let's use: X = normal1, Y = normal2, Z = normal1 × normal2
        # But we need to ensure they form a proper basis
        
        # Recompute to ensure orthogonality
        axis1 = normal1
        axis2 = normal2 - np.dot(normal2, axis1) * axis1
        axis2 = axis2 / np.linalg.norm(axis2)
        axis3 = np.cross(axis1, axis2)
        axis3 = axis3 / np.linalg.norm(axis3)
        
        # Find edge points (points near both planes)
        edge_points = self._find_edge_points(plane1, plane2, points)
        
        if len(edge_points) == 0:
            rospy.logwarn("No edge points found, using plane intersection")
            # Fallback: compute edge line from plane intersection
            edge_center = self._compute_plane_intersection(plane1, plane2)
        else:
            edge_center = np.mean(edge_points, axis=0)
        
        # Calculate translation: move inward from edge along the third axis
        # The third axis corresponds to the smallest dimension
        translation = edge_center - axis3 * (self.dimensions[0] / 2.0)
        
        # Build rotation matrix
        rotation_matrix = np.column_stack([axis1, axis2, axis3])
        
        # Convert to quaternion
        quaternion = tf.transformations.quaternion_from_matrix(
            np.vstack([np.column_stack([rotation_matrix, translation]), [0, 0, 0, 1]])
        )
        
        return {
            'position': translation,
            'orientation': quaternion,
            'rotation_matrix': rotation_matrix,
            'success': True,
            'face_count': 2
        }
    
    def _validate_normal_direction(self, normal, points):
        """
        Ensure normal points outward (toward camera/away from object center)
        
        Args:
            normal: Plane normal vector
            points: Points on the plane
        
        Returns:
            corrected_normal: Normal vector pointing in correct direction
        """
        # Estimate object center (mean of points)
        center = np.mean(points, axis=0)
        
        # For a visible face, the normal should point toward the camera
        # Since we're in camera frame, camera is roughly at origin
        # Normal should point away from center (toward origin)
        camera_direction = -center  # Direction from center to camera
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        
        # If normal points away from camera, flip it
        if np.dot(normal, camera_direction) < 0:
            normal = -normal
        
        return normal
    
    def _identify_face_dimension(self, axis_x, axis_y, inlier_points):
        """
        Identify which cuboid dimension the plane normal corresponds to
        
        Args:
            axis_x: X axis on the plane
            axis_y: Y axis on the plane
            inlier_points: Points on the visible face
        
        Returns:
            int: Index in dimensions array (0=smallest, 1=middle, 2=largest)
        """
        # Calculate area of visible face using bounding box
        # Project points onto plane axes
        centered_points = inlier_points - np.mean(inlier_points, axis=0)
        
        x_coords = np.dot(centered_points, axis_x)
        y_coords = np.dot(centered_points, axis_y)
        
        # Bounding box dimensions
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        area = width * height
        
        # Compare with expected areas for each face
        # Face areas: L×W, L×H, W×H
        expected_areas = [
            self.dimensions[0] * self.dimensions[1],  # smallest × middle
            self.dimensions[0] * self.dimensions[2],  # smallest × largest
            self.dimensions[1] * self.dimensions[2]   # middle × largest
        ]
        
        # Also check edge lengths
        expected_widths = [
            max(self.dimensions[0], self.dimensions[1]),
            max(self.dimensions[0], self.dimensions[2]),
            max(self.dimensions[1], self.dimensions[2])
        ]
        expected_heights = [
            min(self.dimensions[0], self.dimensions[1]),
            min(self.dimensions[0], self.dimensions[2]),
            min(self.dimensions[1], self.dimensions[2])
        ]
        
        # Try all three hypotheses and find best match
        best_match = 0
        best_score = float('inf')
        
        for i in range(3):
            # Score based on area and edge length matching
            area_error = abs(area - expected_areas[i]) / expected_areas[i]
            
            # Check if width/height match expected (account for orientation)
            width_match = min(
                abs(width - expected_widths[i]) / expected_widths[i],
                abs(width - expected_heights[i]) / expected_heights[i]
            )
            height_match = min(
                abs(height - expected_widths[i]) / expected_widths[i],
                abs(height - expected_heights[i]) / expected_heights[i]
            )
            
            score = area_error + width_match + height_match
            
            if score < best_score:
                best_score = score
                best_match = i
        
        # best_match is the index of the missing dimension (the one the normal corresponds to)
        # If best_match=0, normal is largest dimension (index 2)
        # If best_match=1, normal is middle dimension (index 1)
        # If best_match=2, normal is smallest dimension (index 0)
        dimension_map = {0: 2, 1: 1, 2: 0}
        
        rospy.loginfo(f"Identified face: area={area:.4f}, expected={expected_areas[best_match]:.4f}, "
                     f"dimension_index={dimension_map[best_match]}")
        
        return dimension_map[best_match]
    
    def _find_edge_points(self, plane1, plane2, points):
        """
        Find points that are near both planes (edge points)
        
        Args:
            plane1: First plane dictionary
            plane2: Second plane dictionary
            points: All points in point cloud
        
        Returns:
            edge_points: Nx3 array of points near the edge
        """
        edge_points = []
        
        for point in points:
            # Distance to plane 1: |ax + by + cz + d| / sqrt(a² + b² + c²)
            # Since normal is normalized, distance is |dot(normal, point - plane_point)|
            dist1 = abs(np.dot(plane1['normal'], point - plane1['point']))
            dist2 = abs(np.dot(plane2['normal'], point - plane2['point']))
            
            # Point is on edge if it's close to both planes
            if dist1 < self.edge_distance_threshold and dist2 < self.edge_distance_threshold:
                edge_points.append(point)
        
        return np.array(edge_points) if len(edge_points) > 0 else np.array([])
    
    def _compute_plane_intersection(self, plane1, plane2):
        """
        Compute the intersection line of two planes and return a point on it
        
        Args:
            plane1: First plane dictionary
            plane2: Second plane dictionary
        
        Returns:
            point: A point on the intersection line
        """
        n1 = plane1['normal']
        n2 = plane2['normal']
        p1 = plane1['point']
        p2 = plane2['point']
        
        # Direction of intersection line: n1 × n2
        direction = np.cross(n1, n2)
        direction = direction / np.linalg.norm(direction)
        
        # Find a point on the intersection line
        # Solve: n1 · (p + t*d) = n1 · p1 and n2 · (p + t*d) = n2 · p2
        # This is a system of equations
        
        # Use the method: find point closest to both planes
        # Project p1 onto plane2 and p2 onto plane1, then average
        
        # Project p1 onto plane2
        dist_p1_to_plane2 = np.dot(n2, p1 - p2)
        p1_proj = p1 - dist_p1_to_plane2 * n2
        
        # Project p2 onto plane1
        dist_p2_to_plane1 = np.dot(n1, p2 - p1)
        p2_proj = p2 - dist_p2_to_plane1 * n1
        
        # Average (point on intersection line)
        edge_point = (p1_proj + p2_proj) / 2.0
        
        return edge_point


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
        
        # Pose publishers (camera_link and lidar_link frames)
        self.pose_camera_pub = rospy.Publisher('/cuboid/pose/camera_link', PoseStamped, queue_size=1)
        self.pose_lidar_pub = rospy.Publisher('/cuboid/pose/lidar_link', PoseStamped, queue_size=1)
        
        # TF2 buffer and listener for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize cuboid pose estimator
        # Get dimensions from ROS parameter if available, otherwise use default
        cuboid_dims = rospy.get_param('~cuboid_dimensions', [0.3, 0.4, 0.6])
        self.pose_estimator = CuboidPoseEstimator(dimensions=cuboid_dims, tf_buffer=self.tf_buffer) if OPEN3D_AVAILABLE else None
        
        # Rate for publishing (disabled - now triggered by detections)
        # self.publish_rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Depth to PointCloud node initialized")
        rospy.loginfo("Waiting for detections, camera_info and TF transforms...")
        rospy.loginfo("Will only process blue-colored objects")
        if OPEN3D_AVAILABLE and self.pose_estimator:
            rospy.loginfo("Cuboid pose estimation enabled")
    
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
            is_blue = (mean_b > mean_r + 100) and (mean_b > mean_g + 100) and (mean_b > 200)
            
            # rospy.loginfo(f"Object RGB: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}, is_blue={is_blue}")
            
            return is_blue
            
        except Exception as e:
            rospy.logwarn(f"Error checking if object is blue: {e}")
            return False
    
    def filter_blue_points(self, points):
        """
        Filter points to keep only those that are perfectly blue-colored
        
        Args:
            points: Nx6 numpy array (x, y, z, r, g, b) with RGB in 0-1 range
        
        Returns:
            filtered_points: Mx6 numpy array containing only blue points
        """
        if points is None or len(points) == 0:
            return points
        
        # Check if points have RGB data
        if points.shape[1] < 6:
            rospy.logwarn("Points do not have RGB data, cannot filter by color")
            return points
        
        # Extract RGB values (they are in 0-1 range, convert to 0-255 for filtering)
        rgb_values = points[:, 3:6] * 255.0
        r = rgb_values[:, 0]
        g = rgb_values[:, 1]
        b = rgb_values[:, 2]
        
        # Filter for perfectly blue points:
        # - Blue > red + 100
        # - Blue > green + 100
        # - Blue > 200
        blue_mask = (b > r + 100) & (b > g + 100) & (b > 200)
        
        filtered_points = points[blue_mask]
        
        num_filtered = len(points) - len(filtered_points)
        if num_filtered > 0:
            rospy.loginfo_throttle(2, f"Filtered out {num_filtered} non-blue points, kept {len(filtered_points)} blue points")
        
        return filtered_points
    
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
            
            # Filter points to keep only perfectly blue-colored ones
            if rgb_array is not None and points.shape[1] == 6:
                points = self.filter_blue_points(points)
                
                if points is None or len(points) == 0:
                    rospy.logwarn_throttle(2, "No blue points remaining after filtering")
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
            
            # Estimate cuboid pose
            if self.pose_estimator is not None:
                # Extract XYZ coordinates (first 3 columns)
                points_xyz = points[:, :3]
                
                # Estimate pose (pass timestamp for TF lookups)
                pose_result = self.pose_estimator.estimate_pose(points_xyz, timestamp=header.stamp)
                
                if pose_result and pose_result.get('success', False):
                    # Publish pose in camera_link frame
                    pose_camera = self._create_pose_stamped(
                        pose_result['position'],
                        pose_result['orientation'],
                        header.stamp,
                        'camera_link'
                    )
                    self.pose_camera_pub.publish(pose_camera)
                    
                    # Transform pose to lidar_link frame and publish
                    pose_lidar = self._transform_pose_to_lidar_frame(pose_camera, header.stamp)
                    if pose_lidar is not None:
                        self.pose_lidar_pub.publish(pose_lidar)
                    
                    rospy.loginfo_throttle(2, f"Published cuboid pose: position={pose_result['position']}, "
                                             f"face_count={pose_result.get('face_count', 'unknown')}")
                else:
                    rospy.logwarn_throttle(2, "Failed to estimate cuboid pose")
            
        except Exception as e:
            rospy.logwarn(f"Error processing detections: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
    
    def _create_pose_stamped(self, position, orientation, stamp, frame_id):
        """
        Create a PoseStamped message from position and orientation
        
        Args:
            position: numpy array [x, y, z]
            orientation: numpy array [x, y, z, w] (quaternion)
            stamp: rospy.Time
            frame_id: string frame ID
        
        Returns:
            PoseStamped message
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        
        pose_msg.pose.orientation.x = float(orientation[0])
        pose_msg.pose.orientation.y = float(orientation[1])
        pose_msg.pose.orientation.z = float(orientation[2])
        pose_msg.pose.orientation.w = float(orientation[3])
        
        return pose_msg
    
    def _transform_pose_to_lidar_frame(self, pose_camera, timestamp):
        """
        Transform a pose from camera_link frame to lidar_link frame
        
        Args:
            pose_camera: PoseStamped in camera_link frame
            timestamp: rospy.Time for transform lookup
        
        Returns:
            PoseStamped in lidar_link frame, or None if transform fails
        """
        try:
            # Lookup transform from camera_link to lidar_link
            transform = self.tf_buffer.lookup_transform(
                'lidar_link',  # target frame
                'camera_link',  # source frame
                timestamp,
                timeout=rospy.Duration(0.1)
            )
            
            # Extract transform components
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # Convert to numpy arrays
            translation = np.array([t.x, t.y, t.z])
            rotation_quat = np.array([r.x, r.y, r.z, r.w])
            
            # Convert to transformation matrix
            T_camera_to_lidar = tf.transformations.quaternion_matrix(rotation_quat)
            T_camera_to_lidar[:3, 3] = translation
            
            # Get pose in camera frame as transformation matrix
            pos = pose_camera.pose.position
            ori = pose_camera.pose.orientation
            T_pose_camera = tf.transformations.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
            T_pose_camera[:3, 3] = [pos.x, pos.y, pos.z]
            
            # Transform: T_pose_lidar = T_camera_to_lidar * T_pose_camera
            T_pose_lidar = T_camera_to_lidar @ T_pose_camera
            
            # Extract position and orientation
            position_lidar = T_pose_lidar[:3, 3]
            quaternion_lidar = tf.transformations.quaternion_from_matrix(T_pose_lidar)
            
            # Create PoseStamped message
            pose_lidar = PoseStamped()
            pose_lidar.header.stamp = timestamp
            pose_lidar.header.frame_id = 'lidar_link'
            pose_lidar.pose.position.x = float(position_lidar[0])
            pose_lidar.pose.position.y = float(position_lidar[1])
            pose_lidar.pose.position.z = float(position_lidar[2])
            pose_lidar.pose.orientation.x = float(quaternion_lidar[0])
            pose_lidar.pose.orientation.y = float(quaternion_lidar[1])
            pose_lidar.pose.orientation.z = float(quaternion_lidar[2])
            pose_lidar.pose.orientation.w = float(quaternion_lidar[3])
            
            return pose_lidar
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF transform lookup failed for pose: {e}")
            return None
        except Exception as e:
            rospy.logwarn(f"Error transforming pose: {e}")
            return None
    
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

