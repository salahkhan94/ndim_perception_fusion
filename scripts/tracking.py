#!/usr/bin/env python3
"""
ROS node for multi-object tracking and fusion.

This node:
- Subscribes to occupancy grid map from SLAM
- Subscribes to detected object poses from vision
- Implements lidar gating to filter hallucinations
- Performs data association using Hungarian algorithm
- Maintains tracked objects with confidence scores
- Updates tracks based on new detections
"""

import rospy
import numpy as np
from scipy.optimize import linear_sum_assignment
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header
import tf.transformations


class TrackedObject:
    """
    Represents a tracked object with pose, confidence, and metadata.
    """
    
    def __init__(self, obj_id, pose, class_id="cuboid"):
        """
        Initialize a tracked object.
        
        Args:
            obj_id: Unique identifier for this track
            pose: geometry_msgs/Pose in map frame
            class_id: Object class identifier (default: "cuboid")
        """
        self.id = obj_id
        self.pose = pose  # geometry_msgs/Pose
        self.class_id = class_id
        self.confidence = 0.5  # Start at 50%
        self.last_seen = rospy.Time.now()
        self.miss_count = 0  # How many frames have we missed it?
    
    def get_position(self):
        """
        Get position as numpy array [x, y, z].
        
        Returns:
            numpy array [x, y, z]
        """
        return np.array([
            self.pose.position.x,
            self.pose.position.y,
            self.pose.position.z
        ])
    
    def update_pose(self, new_pose, alpha=0.2):
        """
        Update pose using exponential moving average (low-pass filter).
        
        Args:
            new_pose: geometry_msgs/Pose (new detection)
            alpha: Smoothing factor (0.2 means 20% new, 80% old)
        """
        # Position update: new_pose = 0.8 * old_pose + 0.2 * detected_pose
        self.pose.position.x = (1.0 - alpha) * self.pose.position.x + alpha * new_pose.position.x
        self.pose.position.y = (1.0 - alpha) * self.pose.position.y + alpha * new_pose.position.y
        self.pose.position.z = (1.0 - alpha) * self.pose.position.z + alpha * new_pose.position.z
        
        # Orientation update: SLERP (spherical linear interpolation) for quaternions
        old_quat = np.array([
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ])
        new_quat = np.array([
            new_pose.orientation.x,
            new_pose.orientation.y,
            new_pose.orientation.z,
            new_pose.orientation.w
        ])
        
        # Normalize quaternions
        old_quat = old_quat / np.linalg.norm(old_quat)
        new_quat = new_quat / np.linalg.norm(new_quat)
        
        # SLERP interpolation
        dot = np.clip(np.dot(old_quat, new_quat), -1.0, 1.0)
        if abs(dot) > 0.9995:  # Quaternions are very close
            result_quat = old_quat
        else:
            theta = np.arccos(abs(dot))
            sin_theta = np.sin(theta)
            w1 = np.sin((1.0 - alpha) * theta) / sin_theta
            w2 = np.sin(alpha * theta) / sin_theta
            if dot < 0:
                new_quat = -new_quat
            result_quat = w1 * old_quat + w2 * new_quat
            result_quat = result_quat / np.linalg.norm(result_quat)
        
        self.pose.orientation.x = result_quat[0]
        self.pose.orientation.y = result_quat[1]
        self.pose.orientation.z = result_quat[2]
        self.pose.orientation.w = result_quat[3]
    
    def increase_confidence(self, increment=0.1):
        """
        Increase confidence score.
        
        Args:
            increment: Amount to increase (default: 0.1)
        """
        self.confidence = min(1.0, self.confidence + increment)
    
    def decrease_confidence(self, decrement=0.05):
        """
        Decrease confidence score.
        
        Args:
            decrement: Amount to decrease (default: 0.05)
        """
        self.confidence = max(0.0, self.confidence - decrement)


class FusionNode:
    """
    Main fusion node that combines lidar occupancy map and vision detections.
    """
    
    def __init__(self):
        """Initialize the fusion node."""
        rospy.init_node('fusion_node', anonymous=True)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.poses_sub = rospy.Subscriber('/cuboids/poses/map', PoseArray, self.poses_callback)
        
        # Publishers
        self.tracked_poses_pub = rospy.Publisher('/tracked_objects/poses', PoseArray, queue_size=1)
        
        # Storage
        self.occupancy_map = None  # nav_msgs/OccupancyGrid
        self.tracked_objects = []  # List of TrackedObject instances
        self.next_id = 0  # Counter for unique track IDs
        
        # Parameters
        self.association_threshold = rospy.get_param('~association_threshold', 1.5)  # meters
        self.max_miss_count = rospy.get_param('~max_miss_count', 5)  # frames
        self.min_confidence = rospy.get_param('~min_confidence', 0.2)  # minimum confidence to keep track
        self.fov_range = rospy.get_param('~fov_range', 0.5)  # meters - tracks within this distance of detections are considered "in view"
        
        rospy.loginfo("FusionNode initialized")
        rospy.loginfo(f"Association threshold: {self.association_threshold}m")
        rospy.loginfo(f"Max miss count: {self.max_miss_count}")
        rospy.loginfo(f"Min confidence: {self.min_confidence}")
        rospy.loginfo(f"FOV range: {self.fov_range}m")
    
    def map_callback(self, msg):
        """
        Callback for occupancy grid map updates.
        
        Args:
            msg: nav_msgs/OccupancyGrid
        """
        self.occupancy_map = msg
        rospy.logdebug("Received occupancy grid map update")
    
    def poses_callback(self, msg):
        """
        Callback for detected object poses.
        
        Args:
            msg: geometry_msgs/PoseArray in map frame
        """
        if self.occupancy_map is None:
            rospy.logwarn_throttle(5, "Occupancy map not available yet, skipping pose processing")
            return
        
        # Extract valid detections (after lidar gating)
        valid_detections = self._lidar_gating(msg.poses)
        
        if len(valid_detections) == 0:
            rospy.logdebug("No valid detections after lidar gating - camera may be looking at empty space or objects are out of view")
            # Don't penalize tracks when there are no detections - they're likely just out of view
            # Only cleanup tracks that have been missing for too long (from previous frames)
            self._cleanup_tracks()
            return
        
        # Perform data association
        matched_pairs, unmatched_detections, unmatched_tracks = self._data_association(valid_detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            track = self.tracked_objects[track_idx]
            detection = valid_detections[det_idx]
            
            # Update pose (low-pass filter: 0.8 * old + 0.2 * new)
            track.update_pose(detection, alpha=0.2)
            
            # Increase confidence
            track.increase_confidence(0.1)
            
            # Reset miss count
            track.miss_count = 0
            track.last_seen = msg.header.stamp
            
            rospy.logdebug(f"Matched track {track.id} with detection {det_idx}")
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = valid_detections[det_idx]
            new_track = TrackedObject(
                obj_id=self.next_id,
                pose=detection,
                class_id="cuboid"
            )
            self.tracked_objects.append(new_track)
            self.next_id += 1
            rospy.loginfo(f"Created new track {new_track.id} at position ({detection.position.x:.2f}, {detection.position.y:.2f}, {detection.position.z:.2f})")
        
        # Increment miss count only for unmatched tracks that are "in view" (near detections)
        # Tracks far from all detections are likely out of FOV and shouldn't be penalized
        for track_idx in unmatched_tracks:
            track = self.tracked_objects[track_idx]
            track_pos = track.get_position()
            
            # Check if track is within FOV range of any detection
            is_in_view = False
            min_distance_to_detection = float('inf')
            
            for detection in valid_detections:
                det_pos = np.array([
                    detection.position.x,
                    detection.position.y,
                    detection.position.z
                ])
                distance = np.linalg.norm(track_pos - det_pos)
                min_distance_to_detection = min(min_distance_to_detection, distance)
                
                if distance <= self.fov_range:
                    is_in_view = True
                    break
            
            if is_in_view:
                # Track is in view but not detected - this is a real miss
                track.miss_count += 1
                track.decrease_confidence()
                rospy.logdebug(f"Track {track.id} in view but not matched, miss_count={track.miss_count}, "
                             f"min_distance={min_distance_to_detection:.2f}m")
            else:
                # Track is out of view - don't penalize it
                rospy.logdebug(f"Track {track.id} out of view (min_distance={min_distance_to_detection:.2f}m > "
                             f"fov_range={self.fov_range}m), not penalizing")
        
        # Cleanup tracks with low confidence or high miss count
        self._cleanup_tracks()
        
        # Publish tracked objects poses
        self._publish_tracked_poses(msg.header.stamp)
        
        # Log current state
        rospy.loginfo_throttle(2, f"Tracking {len(self.tracked_objects)} objects: "
                                   f"{len(matched_pairs)} matched, "
                                   f"{len(unmatched_detections)} new, "
                                   f"{len(unmatched_tracks)} unmatched")
    
    def _lidar_gating(self, detections):
        """
        Filter detections using occupancy grid map (lidar gating).
        
        Logic:
        - If map value is 0 (Free Space): Check 8 neighbors, REJECT if all are empty
        - If map value is 100 (Occupied): KEEP (Lidar agrees something is there)
        - If map value is -1 (Unknown): KEEP (Camera might see into a shadow the Lidar missed)
        
        Args:
            detections: List of geometry_msgs/Pose objects
            
        Returns:
            List of valid detections (geometry_msgs/Pose)
        """
        if self.occupancy_map is None:
            return []
        
        valid_detections = []
        
        # Get map parameters
        map_data = np.array(self.occupancy_map.data).reshape(
            self.occupancy_map.info.height,
            self.occupancy_map.info.width
        )
        resolution = self.occupancy_map.info.resolution
        origin_x = self.occupancy_map.info.origin.position.x
        origin_y = self.occupancy_map.info.origin.position.y
        
        for detection in detections:
            # Convert global (x, y) to map grid index (r, c)
            x = detection.position.x
            y = detection.position.y
            
            # Convert to grid coordinates
            col = int((x - origin_x) / resolution)
            row = int((y - origin_y) / resolution)
            
            # Check bounds
            if row < 0 or row >= self.occupancy_map.info.height or \
               col < 0 or col >= self.occupancy_map.info.width:
                rospy.logdebug(f"Detection at ({x:.2f}, {y:.2f}) is outside map bounds")
                continue
            
            # Get map value at this cell
            map_value = map_data[row, col]
            
            # Decision logic
            if map_value == 0:  # Free Space
                # Check 8 neighbors
                neighbors_empty = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr = row + dr
                        nc = col + dc
                        if 0 <= nr < self.occupancy_map.info.height and \
                           0 <= nc < self.occupancy_map.info.width:
                            if map_data[nr, nc] != 0:  # Not empty
                                neighbors_empty = False
                                break
                    if not neighbors_empty:
                        break
                
                if neighbors_empty:
                    # REJECT: All neighbors are empty, likely hallucination
                    rospy.logdebug(f"Rejected detection at ({x:.2f}, {y:.2f}): Free space with empty neighbors")
                    continue
                else:
                    # KEEP: Some neighbors are not empty
                    valid_detections.append(detection)
            
            elif map_value == 100:  # Occupied
                # KEEP: Lidar agrees something is there
                valid_detections.append(detection)
            
            elif map_value == -1:  # Unknown
                # KEEP: Camera might see into a shadow the Lidar missed
                valid_detections.append(detection)
            
            else:
                # Intermediate probability values: KEEP if > 50, REJECT otherwise
                if map_value > 50:
                    valid_detections.append(detection)
                else:
                    rospy.logdebug(f"Rejected detection at ({x:.2f}, {y:.2f}): Low occupancy probability ({map_value})")
        
        return valid_detections
    
    def _data_association(self, detections):
        """
        Perform data association using Hungarian algorithm.
        
        Args:
            detections: List of geometry_msgs/Pose objects (valid detections)
            
        Returns:
            matched_pairs: List of (track_idx, det_idx) tuples
            unmatched_detections: List of detection indices that don't match any track
            unmatched_tracks: List of track indices that don't match any detection
        """
        if len(self.tracked_objects) == 0:
            # No existing tracks, all detections are new
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            # No detections, all tracks are unmatched
            return [], [], list(range(len(self.tracked_objects)))
        
        # Create cost matrix: rows = tracks, columns = detections
        n_tracks = len(self.tracked_objects)
        n_detections = len(detections)
        cost_matrix = np.zeros((n_tracks, n_detections))
        
        # Fill cost matrix with Euclidean distances
        for i, track in enumerate(self.tracked_objects):
            track_pos = track.get_position()
            for j, detection in enumerate(detections):
                det_pos = np.array([
                    detection.position.x,
                    detection.position.y,
                    detection.position.z
                ])
                # Euclidean distance
                distance = np.linalg.norm(track_pos - det_pos)
                cost_matrix[i, j] = distance
        
        # Hungarian algorithm to find optimal assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matched_pairs = []
        matched_track_set = set()
        matched_det_set = set()
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            distance = cost_matrix[track_idx, det_idx]
            if distance <= self.association_threshold:
                matched_pairs.append((track_idx, det_idx))
                matched_track_set.add(track_idx)
                matched_det_set.add(det_idx)
            else:
                rospy.logdebug(f"Rejected match: track {track_idx} <-> det {det_idx}, distance={distance:.3f}m > threshold={self.association_threshold}m")
        
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(n_detections) if i not in matched_det_set]
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_set]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _cleanup_tracks(self):
        """
        Remove tracks with low confidence or high miss count.
        """
        tracks_to_remove = []
        for i, track in enumerate(self.tracked_objects):
            if track.confidence < self.min_confidence or track.miss_count >= self.max_miss_count:
                tracks_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(tracks_to_remove):
            track = self.tracked_objects[i]
            rospy.loginfo(f"Removing track {track.id}: confidence={track.confidence:.2f}, miss_count={track.miss_count}")
            self.tracked_objects.pop(i)
    
    def _publish_tracked_poses(self, stamp=None):
        """
        Publish all tracked object poses as a PoseArray.
        
        Args:
            stamp: rospy.Time for the message header (default: current time)
        """
        if stamp is None:
            stamp = rospy.Time.now()
        
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = "map"  # All poses are in map frame
        
        # Add poses from all tracked objects
        for track in self.tracked_objects:
            pose_array.poses.append(track.pose)
        
        # Publish
        self.tracked_poses_pub.publish(pose_array)
        
        rospy.logdebug(f"Published {len(pose_array.poses)} tracked object poses")
    
    def get_tracked_objects(self):
        """
        Get list of currently tracked objects.
        
        Returns:
            List of TrackedObject instances
        """
        return self.tracked_objects.copy()


if __name__ == '__main__':
    try:
        node = FusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

