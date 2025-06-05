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
import tf2_geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TransformStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs.point_cloud2 as pc2
import tf_conversions

from ekf_slam.ekf_slam import (
    RangeBearingKFSLAM2D, 
    RangeBearingObservation, 
    SensoryFrame, 
    ActionCollection,
    EKFSLAMOptions
)
from ekf_slam.data_association import DataAssociation, PerceptionProcessor


class EKFSLAMNode:
    """
    ROS node for EKF SLAM system.
    """
    
    def __init__(self):
        """Initialize the EKF SLAM node."""
        rospy.init_node('ekf_slam_node', anonymous=True)
        
        # Frame IDs
        self.global_frame_id = rospy.get_param('~global_frame_id', 'map')
        self.odom_frame_id = rospy.get_param('~odom_frame_id', 'odom')
        self.base_frame_id = rospy.get_param('~base_frame_id', 'base_link')
        self.lidar_frame_id = rospy.get_param('~lidar_frame_id', 'lidar_link')

        # EKF SLAM parameters
        process_noise_std = rospy.get_param('~process_noise_std', [0.02, 0.02, 0.01])
        sensor_range_std = rospy.get_param('~sensor_range_std', 0.05)
        sensor_bearing_std = rospy.get_param('~sensor_bearing_std', 0.02)
        
        # Data association parameters
        max_assoc_dist = rospy.get_param('~max_association_distance', 2.0)
        mahal_threshold = rospy.get_param('~mahalanobis_threshold', 9.21)
        
        # Perception parameters
        min_cluster_size = rospy.get_param('~min_cluster_size', 5)
        max_cluster_size = rospy.get_param('~max_cluster_size', 50)
        cluster_threshold = rospy.get_param('~cluster_distance_threshold', 0.5)
        max_sensor_range = rospy.get_param('~max_sensor_range', 20.0)
        
        # Initialize EKF SLAM
        self.slam = RangeBearingKFSLAM2D()
        
        # Configure SLAM options
        options = EKFSLAMOptions()
        options.std_q_no_odo = process_noise_std
        options.std_sensor_range = sensor_range_std
        options.std_sensor_yaw = sensor_bearing_std
        options.create_simplemap = True
        self.slam.options = options
        
        # Initialize data association and perception
        self.data_association = DataAssociation(
            max_association_distance=max_assoc_dist,
            mahalanobis_threshold=mahal_threshold
        )
        
        self.perception = PerceptionProcessor(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_distance_threshold=cluster_threshold
        )
        
        self.max_sensor_range = max_sensor_range
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # State tracking
        self.last_odom = None
        self.last_odom_time = None
        
        # Publishers
        self.pose_pub = rospy.Publisher(
            '~robot_pose', PoseWithCovarianceStamped, queue_size=1
        )
        self.landmarks_pub = rospy.Publisher(
            '~landmarks', MarkerArray, queue_size=1
        )
        
        # Subscribers
        self.odom_sub = rospy.Subscriber(
            '/odom', Odometry, self.odom_callback, queue_size=1
        )
        self.pointcloud_sub = rospy.Subscriber(
            '/lidar/clustered', PointCloud2, self.pointcloud_callback, queue_size=1
        )
        
        rospy.loginfo("EKF SLAM node initialized")

    def odom_callback(self, msg):
        """Handle odometry messages for motion model."""
        try:
            current_time = msg.header.stamp
            
            if self.last_odom is not None:
                # Compute odometry increment
                dt = (current_time - self.last_odom_time).to_sec()
                
                if dt > 0 and dt < 1.0:  # Reasonable time step
                    # Extract poses
                    curr_x = msg.pose.pose.position.x
                    curr_y = msg.pose.pose.position.y
                    curr_yaw = tf_conversions.transformations.euler_from_quaternion([
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w
                    ])[2]
                    
                    prev_x = self.last_odom.pose.pose.position.x
                    prev_y = self.last_odom.pose.pose.position.y
                    prev_yaw = tf_conversions.transformations.euler_from_quaternion([
                        self.last_odom.pose.pose.orientation.x,
                        self.last_odom.pose.pose.orientation.y,
                        self.last_odom.pose.pose.orientation.z,
                        self.last_odom.pose.pose.orientation.w
                    ])[2]
                    
                    # Compute increments
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    dyaw = self._wrap_angle(curr_yaw - prev_yaw)
                    
                    # Create action
                    odom_cov = np.array(msg.pose.covariance).reshape(6, 6)[:3, :3]
                    action = ActionCollection(
                        dx=dx, dy=dy, dyaw=dyaw,
                        timestamp=current_time.to_sec(),
                        covariance=odom_cov
                    )
                    
                    # Store for next iteration
                    self.last_action = action
                    
                    rospy.logdebug(f"Odometry increment: dx={dx:.3f}, dy={dy:.3f}, dyaw={dyaw:.3f}")
            
            # Update last odometry
            self.last_odom = msg
            self.last_odom_time = current_time
            
        except Exception as e:
            rospy.logwarn(f"Error processing odometry: {e}")

    def pointcloud_callback(self, msg):
        """Handle point cloud messages for landmark detection."""
        try:
            # Extract points from point cloud
            points = self._extract_points_from_pointcloud(msg)
            
            if len(points) == 0:
                return
            
            # Process point cloud to extract measurements
            measurements = self.perception.process_point_cloud(points)
            
            if not measurements:
                return
            
            # Convert measurements to range-bearing observations
            observations = []
            for i, (range_val, bearing) in enumerate(measurements):
                obs = RangeBearingObservation(
                    range=range_val,
                    yaw=bearing,
                    pitch=0.0,
                    landmark_id=-1  # Unknown landmark
                )
                observations.append(obs)
            
            # Create sensor frame
            sensor_frame = SensoryFrame(
                observations=observations,
                timestamp=msg.header.stamp.to_sec(),
                sensor_pose=self._get_sensor_pose(msg.header.frame_id)
            )
            
            # Process with EKF SLAM if we have motion data
            if hasattr(self, 'last_action'):
                self.slam.process_action_observation(self.last_action, sensor_frame)
                
                # Publish results
                self._publish_robot_pose()
                self._publish_landmarks()
                self._publish_tf()
                
                rospy.logdebug(f"Processed {len(observations)} observations")
            else:
                rospy.logdebug("Waiting for odometry data...")
                
        except Exception as e:
            rospy.logwarn(f"Error processing point cloud: {e}")

    def _extract_points_from_pointcloud(self, pointcloud_msg):
        """Extract 2D points from PointCloud2 message."""
        points = []
        
        try:
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point
                
                # Filter points (remove ground, limit range)
                if abs(z) < 0.5 and np.sqrt(x**2 + y**2) < self.max_sensor_range:
                    points.append([x, y])
        
        except Exception as e:
            rospy.logwarn(f"Error extracting points: {e}")
        
        return np.array(points) if points else np.array([]).reshape(0, 2)

    def _get_sensor_pose(self, frame_id):
        """Get sensor pose relative to base_link."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame_id, frame_id, rospy.Time(0), rospy.Duration(1.0)
            )
            
            # Extract pose
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # Convert quaternion to euler
            roll, pitch, yaw = tf_conversions.transformations.euler_from_quaternion([r.x, r.y, r.z, r.w])
            
            return np.array([t.x, t.y, t.z, roll, pitch, yaw])
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get sensor pose: {e}")
            return np.zeros(6)  # Assume sensor at robot origin

    def _publish_robot_pose(self):
        """Publish robot pose estimate."""
        try:
            pose_pdf = self.slam.get_current_robot_pose()
            
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.global_frame_id
            
            # Fill pose
            pose_msg.pose.pose.position.x = pose_pdf.mean.x
            pose_msg.pose.pose.position.y = pose_pdf.mean.y
            pose_msg.pose.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            q = tf_conversions.transformations.quaternion_from_euler(0, 0, pose_pdf.mean.yaw)
            pose_msg.pose.pose.orientation.x = q[0]
            pose_msg.pose.pose.orientation.y = q[1]
            pose_msg.pose.pose.orientation.z = q[2]
            pose_msg.pose.pose.orientation.w = q[3]
            
            # Fill covariance (6x6 matrix, but we only have 3x3)
            covariance = np.zeros((6, 6))
            covariance[:2, :2] = pose_pdf.cov[:2, :2]  # x, y
            covariance[5, 5] = pose_pdf.cov[2, 2]  # yaw
            pose_msg.pose.covariance = covariance.flatten().tolist()
            
            self.pose_pub.publish(pose_msg)
            
        except Exception as e:
            rospy.logwarn(f"Error publishing robot pose: {e}")

    def _publish_landmarks(self):
        """Publish landmark estimates as markers."""
        try:
            _, landmarks, landmark_ids, _, _ = self.slam.get_current_state()
            
            marker_array = MarkerArray()
            
            for i in range(len(landmarks)):
                marker = Marker()
                marker.header.frame_id = self.global_frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "landmarks"
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                # Position
                marker.pose.position.x = landmarks[i][0]
                marker.pose.position.y = landmarks[i][1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Scale
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 1.0
                
                # Color based on landmark ID
                if i in landmark_ids:
                    # Known landmark - green
                    marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
                else:
                    # New landmark - red
                    marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
                
                marker_array.markers.append(marker)
            
            # Add delete marker for unused IDs
            for i in range(len(landmarks), 100):  # Clear up to 100 old markers
                marker = Marker()
                marker.header.frame_id = self.global_frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "landmarks"
                marker.id = i
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
            
            self.landmarks_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logwarn(f"Error publishing landmarks: {e}")

    def _publish_tf(self):
        """Publish TF transform from map to odom."""
        try:
            # Get robot pose in map frame
            pose_pdf = self.slam.get_current_robot_pose()
            
            # For now, publish identity transform
            # In a full implementation, you'd compute the map->odom transform
            # based on the difference between SLAM estimate and odometry
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = self.global_frame_id
            transform.child_frame_id = self.odom_frame_id
            
            # Identity transform (assume map and odom are aligned)
            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 0.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish TF: {e}")

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi


def main():
    """Main function."""
    try:
        node = EKFSLAMNode()
        rospy.loginfo("EKF SLAM node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("EKF SLAM node interrupted")
    except Exception as e:
        rospy.logerr(f"EKF SLAM node failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
