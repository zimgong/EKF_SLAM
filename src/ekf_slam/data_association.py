#!/usr/bin/env python3
"""
Data association system using KDTree for MR24 Driverless.
"""

import numpy as np
import rospy
from scipy.spatial import KDTree
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


class DataAssociation:
    """
    Data association system using KDTree for efficient landmark matching.
    """
    
    def __init__(self, max_association_distance=2.0, mahalanobis_threshold=9.21):
        """
        Initialize data association system.
        
        Args:
            max_association_distance: Maximum Euclidean distance for association
            mahalanobis_threshold: Chi-squared threshold for Mahalanobis distance (95% confidence)
        """
        self.max_association_distance = max_association_distance
        self.mahalanobis_threshold = mahalanobis_threshold
        
        # KDTree for efficient nearest neighbor search
        self.kdtree = None
        self.landmark_positions = []
        self.landmark_ids = []
        
        rospy.loginfo("Data association system initialized")
    
    def update_landmarks(self, landmarks):
        """
        Update the KDTree with current landmark estimates.
        
        Args:
            landmarks: Dictionary {landmark_id: [x, y]} of landmark positions
        """
        if not landmarks:
            self.kdtree = None
            self.landmark_positions = []
            self.landmark_ids = []
            return
        
        # Extract positions and IDs
        self.landmark_ids = list(landmarks.keys())
        self.landmark_positions = [landmarks[lid] for lid in self.landmark_ids]
        
        # Build KDTree
        if len(self.landmark_positions) > 0:
            self.kdtree = KDTree(np.array(self.landmark_positions))
    
    def associate_measurements(self, measurements, robot_pose, robot_covariance, 
                             measurement_covariance):
        """
        Associate measurements with existing landmarks or mark as new.
        
        Args:
            measurements: List of [range, bearing] measurements
            robot_pose: Current robot pose [x, y, theta]
            robot_covariance: Robot pose covariance matrix (3x3)
            measurement_covariance: Measurement covariance matrix (2x2)
        
        Returns:
            associations: List of (landmark_id, measurement) for existing landmarks
            new_measurements: List of measurements for new landmarks
        """
        if not measurements:
            return [], []
        
        # Convert measurements to Cartesian coordinates
        cartesian_measurements = self._polar_to_cartesian(measurements, robot_pose)
        
        associations = []
        new_measurements = []
        used_landmarks = set()
        
        for i, (measurement, cart_pos) in enumerate(zip(measurements, cartesian_measurements)):
            # Find potential associations
            candidates = self._find_candidates(cart_pos)
            
            # Validate associations using Mahalanobis distance
            best_match = self._validate_association(
                measurement, cart_pos, candidates, robot_pose, 
                robot_covariance, measurement_covariance, used_landmarks
            )
            
            if best_match is not None:
                landmark_id = best_match
                associations.append((landmark_id, measurement))
                used_landmarks.add(landmark_id)
                rospy.logdebug(f"Associated measurement {i} with landmark {landmark_id}")
            else:
                new_measurements.append(measurement)
                rospy.logdebug(f"Measurement {i} marked as new landmark")
        
        return associations, new_measurements
    
    def _polar_to_cartesian(self, measurements, robot_pose):
        """Convert polar measurements to Cartesian coordinates."""
        x_robot, y_robot, theta_robot = robot_pose
        cartesian_positions = []
        
        for range_meas, bearing_meas in measurements:
            absolute_bearing = theta_robot + bearing_meas
            x = x_robot + range_meas * np.cos(absolute_bearing)
            y = y_robot + range_meas * np.sin(absolute_bearing)
            cartesian_positions.append([x, y])
        
        return cartesian_positions
    
    def _find_candidates(self, measurement_position):
        """Find candidate landmarks using KDTree."""
        if self.kdtree is None or len(self.landmark_positions) == 0:
            return []
        
        # Query KDTree for nearby landmarks
        distances, indices = self.kdtree.query(
            [measurement_position], 
            k=min(3, len(self.landmark_positions)),  # Find up to 3 nearest neighbors
            return_distance=True
        )
        
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= self.max_association_distance:
                landmark_id = self.landmark_ids[idx]
                candidates.append((landmark_id, dist))
        
        # Sort by distance
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def _validate_association(self, measurement, measurement_position, candidates, 
                            robot_pose, robot_covariance, measurement_covariance, 
                            used_landmarks):
        """
        Validate association using Mahalanobis distance.
        
        Args:
            measurement: [range, bearing] measurement
            measurement_position: [x, y] Cartesian position of measurement
            candidates: List of (landmark_id, distance) candidates
            robot_pose: Robot pose [x, y, theta]
            robot_covariance: Robot covariance matrix
            measurement_covariance: Measurement covariance matrix
            used_landmarks: Set of already used landmark IDs
        
        Returns:
            Best matching landmark ID or None
        """
        if not candidates:
            return None
        
        x_robot, y_robot, theta_robot = robot_pose
        range_meas, bearing_meas = measurement
        
        best_match = None
        min_mahalanobis = float('inf')
        
        for landmark_id, euclidean_dist in candidates:
            # Skip if landmark already used
            if landmark_id in used_landmarks:
                continue
            
            # Get landmark position
            landmark_idx = self.landmark_ids.index(landmark_id)
            landmark_pos = self.landmark_positions[landmark_idx]
            x_landmark, y_landmark = landmark_pos
            
            # Predicted measurement
            dx = x_landmark - x_robot
            dy = y_landmark - y_robot
            predicted_range = np.sqrt(dx**2 + dy**2)
            predicted_bearing = np.arctan2(dy, dx) - theta_robot
            predicted_bearing = self._normalize_angle(predicted_bearing)
            
            # Innovation
            innovation = np.array([
                range_meas - predicted_range,
                self._normalize_angle(bearing_meas - predicted_bearing)
            ])
            
            # Innovation covariance (simplified)
            # In practice, this should include the full Jacobian and state covariance
            S = measurement_covariance + np.eye(2) * 0.1  # Add some uncertainty
            
            # Mahalanobis distance
            try:
                mahal_dist = innovation.T @ np.linalg.inv(S) @ innovation
                
                if mahal_dist < self.mahalanobis_threshold and mahal_dist < min_mahalanobis:
                    min_mahalanobis = mahal_dist
                    best_match = landmark_id
                    
            except np.linalg.LinAlgError:
                # Skip if covariance matrix is singular
                continue
        
        return best_match
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_association_stats(self):
        """Get statistics about the association system."""
        return {
            'num_landmarks': len(self.landmark_ids),
            'max_distance': self.max_association_distance,
            'mahalanobis_threshold': self.mahalanobis_threshold
        }


class PerceptionProcessor:
    """
    Process point cloud data to extract landmark measurements.
    """
    
    def __init__(self, min_cluster_size=5, max_cluster_size=50, 
                 cluster_distance_threshold=0.5):
        """
        Initialize perception processor.
        
        Args:
            min_cluster_size: Minimum points in a cluster to be considered a landmark
            max_cluster_size: Maximum points in a cluster
            cluster_distance_threshold: Distance threshold for clustering
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.cluster_distance_threshold = cluster_distance_threshold
        
        rospy.loginfo("Perception processor initialized")
    
    def process_point_cloud(self, points):
        """
        Process point cloud to extract landmark measurements.
        
        Args:
            points: Nx2 array of [x, y] points in robot frame
        
        Returns:
            measurements: List of [range, bearing] measurements
        """
        if len(points) == 0:
            return []
        
        # Simple clustering based on distance
        clusters = self._cluster_points(points)
        
        # Extract measurements from clusters
        measurements = []
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                # Use centroid of cluster as landmark position
                centroid = np.mean(cluster, axis=0)
                range_val = np.linalg.norm(centroid)
                bearing = np.arctan2(centroid[1], centroid[0])
                measurements.append([range_val, bearing])
        
        return measurements
    
    def _cluster_points(self, points):
        """Simple distance-based clustering."""
        if len(points) == 0:
            return []
        
        clusters = []
        remaining_points = points.copy()
        
        while len(remaining_points) > 0:
            # Start new cluster with first remaining point
            cluster = [remaining_points[0]]
            remaining_points = remaining_points[1:]
            
            # Add nearby points to cluster
            i = 0
            while i < len(remaining_points) and len(cluster) < self.max_cluster_size:
                point = remaining_points[i]
                
                # Check if point is close to any point in current cluster
                min_dist = min(np.linalg.norm(point - cp) for cp in cluster)
                
                if min_dist <= self.cluster_distance_threshold:
                    cluster.append(point)
                    remaining_points = np.delete(remaining_points, i, axis=0)
                else:
                    i += 1
            
            clusters.append(np.array(cluster))
        
        return clusters 
