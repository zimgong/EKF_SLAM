#!/usr/bin/env python3
"""
EKF SLAM ROS node for MR24 Driverless.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs.point_cloud2 as pc2

from ekf_slam.ekf_slam import EKFSlam
from ekf_slam.data_association import DataAssociation, PerceptionProcessor


class EKFSLAMNode:
    """
    ROS node for Mapping system.
    """
    
    def __init__(self):
        """Initialize the EKF SLAM node."""
        rospy.init_node('ekf_slam_node', anonymous=True)
        
        self.global_frame_id = rospy.get_param('~global_frame_id', 'map')
        self.odom_frame_id = rospy.get_param('~odom_frame_id', 'odom')
        self.base_frame_id = rospy.get_param('~base_frame_id', 'base_link')

        self.landmark_poses_ = {} # landmark poses with respect to the map
        
        # EKF parameters
        process_noise = rospy.get_param('~process_noise_std', 0.1)
        measurement_noise = rospy.get_param('~measurement_noise_std', 0.5)
        
        # Data association parameters
        max_assoc_dist = rospy.get_param('~max_association_distance', 2.0)
        mahal_threshold = rospy.get_param('~mahalanobis_threshold', 9.21)
        
        # Perception parameters
        min_cluster_size = rospy.get_param('~min_cluster_size', 5)
        max_cluster_size = rospy.get_param('~max_cluster_size', 50)
        cluster_threshold = rospy.get_param('~cluster_distance_threshold', 0.5)
        
        # Initialize components
        self.ekf = EKFSlam(
            process_noise_std=process_noise,
            measurement_noise_std=measurement_noise
        )
        
        self.data_association = DataAssociation(
            max_association_distance=max_assoc_dist,
            mahalanobis_threshold=mahal_threshold
        )
        
        self.perception = PerceptionProcessor(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_distance_threshold=cluster_threshold
        )
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Publishers
        self.pose_pub = rospy.Publisher(
            '~robot_pose', PoseWithCovarianceStamped, queue_size=1
        )
        self.landmarks_pub = rospy.Publisher(
            '~landmarks', MarkerArray, queue_size=1
        )
        self.map_pub = rospy.Publisher(
            '~occupancy_grid', OccupancyGrid, queue_size=1
        )
        
        # Subscribers
        self.pointcloud_sub = rospy.Subscriber(
            '/lidar/clustered', PointCloud2, self.landmark_callback, queue_size=1
        )
        
        rospy.loginfo("EKF SLAM node initialized")

    def update_sensor_pose(self, frame_id: str) -> None:
        """Update sensor pose with respect to the map."""
        try:
            timeout = rospy.Duration(1.0)
            transform = self.tf_buffer.lookup_transform(
                self.base_frame_id, frame_id, rospy.Time(0), timeout)
        except tf2_ros.TransformException as e:
            rospy.logwarn(
                f"Failed to get transform target_frame (%s) to source_frame (%s): %s",
                self.base_frame_id, frame_id, e)
            return
        self.landmark_poses_[frame_id] = transform
        return

    def wait_for_transform(self, target_frame: str, source_frame: str, time: rospy.Time, timeout: rospy.Duration) -> tf2_ros.TransformStamped:
        """Wait for transform between two frames."""
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, time, timeout)
        except (tf2_ros.TransformException) as e:
            rospy.logwarn(
                f"Failed to get transform target_frame (%s) to source_frame (%s): %s", 
                target_frame, source_frame, e)
            return
        return transform

    def odom_callback(self, msg):
        """Handle odometry messages for motion model."""
        return self.wait_for_transform(self.odom_frame_id, self.base_frame_id, msg.header.stamp, rospy.Duration(1))

    def landmark_callback(self, msg):
        """Handle point cloud messages for landmark detection."""
        if msg.header.frame_id not in self.landmark_poses_:
            self.update_sensor_pose(msg.header.frame_id)
        else:
            pose = self.landmark_poses_[msg.header.frame_id]
            # Transform point cloud to robot frame
            points = self.extract_points_from_pointcloud(msg, pose)
            
            if len(points) == 0:
                return
            
            # Process point cloud to extract measurements
            measurements = self.perception.process_point_cloud(points)
            
            if not measurements:
                return
            
            # Get current robot pose estimate
            robot_pose = self.ekf.get_robot_pose()
            robot_covariance = self.ekf.get_robot_covariance()
            
            # Measurement covariance (simplified)
            measurement_covariance = np.eye(2) * self.ekf.measurement_noise_std**2
            
            # Update data association with current landmarks
            landmarks = self.ekf.get_landmarks()
            self.data_association.update_landmarks(landmarks)
            
            # Associate measurements
            associations, new_measurements = self.data_association.associate_measurements(
                measurements, robot_pose, robot_covariance, measurement_covariance
            )
            
            # Update EKF with associated measurements
            if associations:
                self.ekf.update(associations)
            
            # Add new landmarks
            for i, measurement in enumerate(new_measurements):
                new_landmark_id = f"landmark_{len(self.ekf.landmarks)}_{rospy.Time.now().to_nsec()}"
                self.ekf.update([(new_landmark_id, measurement)])
            
            rospy.logdebug(f"Processed {len(measurements)} measurements, "
                          f"{len(associations)} associations, "
                          f"{len(new_measurements)} new landmarks")

            self.viz_state()
            self.viz_data_association()
            self.publish_tf()
    
    def extract_points_from_pointcloud(self, pointcloud_msg, pose):
        """Extract 2D points from PointCloud2 message."""
        
        # Extract points
        points = []
        for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            
            # Filter points (e.g., remove ground, limit range)
            if abs(z) < 0.5 and np.sqrt(x**2 + y**2) < 20.0:  # Simple filtering
                if pose is not None:
                    # Apply transform if needed
                    # For simplicity, assuming 2D transformation
                    x, y = pose.transform_point(np.array([x, y, 0]))
                points.append([x, y])
        
        return np.array(points)
    
    def viz_state(self):
        """Publish robot pose estimate."""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.global_frame_id
        
        # Get pose estimate
        robot_pose = self.ekf.get_robot_pose()
        robot_covariance = self.ekf.get_robot_covariance()
        
        # Fill pose
        pose_msg.pose.pose.position.x = robot_pose[0]
        pose_msg.pose.pose.position.y = robot_pose[1]
        pose_msg.pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw = robot_pose[2]
        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = np.sin(yaw / 2.0)
        pose_msg.pose.pose.orientation.w = np.cos(yaw / 2.0)
        
        # Fill covariance (6x6 matrix, but we only have 3x3)
        covariance = np.zeros((6, 6))
        covariance[:2, :2] = robot_covariance[:2, :2]  # x, y
        covariance[5, 5] = robot_covariance[2, 2]  # yaw
        pose_msg.pose.covariance = covariance.flatten().tolist()
        
        self.pose_pub.publish(pose_msg)
    
    def viz_data_association(self):
        """Publish landmark estimates as markers."""
        ma = MarkerArray()
        da = self.data_association.get_data_association()
        
        for i, (landmark_id, position) in enumerate(da.results.associations.items()):
            marker = Marker()
            marker.header.frame_id = self.global_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 1.0
            
            # Color
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # Red
            
            ma.markers.append(marker)
        
        self.landmarks_pub.publish(ma)
    
    def publish_tf(self):
        """Publish TF transform from map to odom."""
        try:
            # Get robot pose in map frame
            robot_pose = self.ekf.get_robot_pose()
            
            # Create transform
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = self.global_frame_id
            transform.child_frame_id = self.odom_frame_id
            
            # For simplicity, assume map and odom are aligned
            # In practice, you'd compute the transform based on the difference
            # between EKF estimate and odometry
            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 0.0
            transform.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish TF: {e}")


def main():
    try:
        node = EKFSLAMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"EKF SLAM node failed: {e}")


if __name__ == '__main__':
    main()
