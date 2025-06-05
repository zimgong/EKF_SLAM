#!/usr/bin/env python3
"""
Example usage of the Python EKF SLAM 2D implementation.

This script demonstrates how to use the RangeBearingKFSLAM2D class
with synthetic sensor data and odometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import math

from ekf_slam import (
    RangeBearingKFSLAM2D, 
    RangeBearingObservation, 
    SensorFrame, 
    OdometryAction,
    EKFSLAMOptions
)


def create_synthetic_landmarks() -> List[Tuple[float, float]]:
    """Create a set of synthetic landmarks in a 2D environment."""
    landmarks = [
        (10.0, 0.0),   # Landmark 1
        (10.0, 10.0),  # Landmark 2  
        (0.0, 10.0),   # Landmark 3
        (-10.0, 10.0), # Landmark 4
        (-10.0, 0.0),  # Landmark 5
        (-10.0, -10.0), # Landmark 6
        (0.0, -10.0),  # Landmark 7
        (10.0, -10.0), # Landmark 8
    ]
    return landmarks


def generate_odometry_sequence() -> List[OdometryAction]:
    """Generate a synthetic odometry sequence (circular path)."""
    actions = []
    dt = 0.1  # time step
    v = 1.0   # linear velocity (m/s)
    w = 0.1   # angular velocity (rad/s)
    
    # Create circular motion
    for i in range(100):
        timestamp = i * dt
        
        # Move forward with slight turning
        dx = v * dt * np.cos(w * timestamp)  
        dy = v * dt * np.sin(w * timestamp)
        dyaw = w * dt
        
        # Add some noise to odometry
        dx += np.random.normal(0, 0.01)
        dy += np.random.normal(0, 0.01) 
        dyaw += np.random.normal(0, 0.01)
        
        # Simple covariance model
        odom_cov = np.diag([0.01**2, 0.01**2, 0.01**2])
        
        action = OdometryAction(
            dx=dx, dy=dy, dyaw=dyaw, 
            timestamp=timestamp,
            covariance=odom_cov
        )
        actions.append(action)
    
    return actions


def simulate_range_bearing_observations(
    robot_pose: np.ndarray, 
    landmarks: List[Tuple[float, float]],
    max_range: float = 15.0,
    timestamp: float = 0.0
) -> SensorFrame:
    """
    Simulate range-bearing observations from current robot pose.
    
    Parameters:
    -----------
    robot_pose : np.ndarray
        Current robot pose [x, y, yaw]
    landmarks : List[Tuple[float, float]]
        True landmark positions
    max_range : float
        Maximum sensor range
    timestamp : float
        Observation timestamp
        
    Returns:
    --------
    SensorFrame
        Simulated sensor observations
    """
    observations = []
    
    robot_x, robot_y, robot_yaw = robot_pose
    
    for lm_id, (lm_x, lm_y) in enumerate(landmarks):
        # Compute relative position
        dx = lm_x - robot_x
        dy = lm_y - robot_y
        
        # Compute true range and bearing
        true_range = np.sqrt(dx**2 + dy**2)
        true_bearing = math.atan2(dy, dx) - robot_yaw
        
        # Normalize bearing to [-π, π]
        true_bearing = ((true_bearing + np.pi) % (2 * np.pi)) - np.pi
        
        # Only observe landmarks within sensor range
        if true_range <= max_range:
            # Add sensor noise
            noise_range = np.random.normal(0, 0.1)  # 10cm std dev
            noise_bearing = np.random.normal(0, 0.05)  # ~3 degree std dev
            
            observed_range = max(0.1, true_range + noise_range)  # Avoid zero range
            observed_bearing = true_bearing + noise_bearing
            
            obs = RangeBearingObservation(
                range=observed_range,
                yaw=observed_bearing, 
                pitch=0.0,
                landmark_id=lm_id  # Known landmark IDs for this example
            )
            observations.append(obs)
    
    return SensorFrame(
        observations=observations,
        timestamp=timestamp,
        sensor_pose=np.zeros(6)  # Sensor at robot origin
    )


def run_ekf_slam_simulation():
    """Run a complete EKF SLAM simulation with synthetic data."""
    
    print("Starting EKF SLAM 2D simulation...")
    
    # Create landmarks and robot trajectory
    true_landmarks = create_synthetic_landmarks()
    odometry_sequence = generate_odometry_sequence()
    
    # Initialize EKF SLAM
    slam = RangeBearingKFSLAM2D()
    
    # Configure options
    options = EKFSLAMOptions()
    options.std_sensor_range = 0.1  # 10cm range noise
    options.std_sensor_yaw = 0.05   # ~3 degree bearing noise
    options.std_q_no_odo = [0.05, 0.05, math.radians(2.0)]  # Process noise
    options.create_simplemap = True
    slam.options = options
    
    # Storage for results
    estimated_poses = []
    estimated_landmarks = []
    true_poses = []
    
    # Simulate robot motion and observations
    true_robot_pose = np.array([0.0, 0.0, 0.0])  # Start at origin
    
    print(f"Processing {len(odometry_sequence)} time steps...")
    
    for i, odom_action in enumerate(odometry_sequence):
        # Update true robot pose (perfect motion model for ground truth)
        true_robot_pose[0] += odom_action.dx
        true_robot_pose[1] += odom_action.dy  
        true_robot_pose[2] += odom_action.dyaw
        true_robot_pose[2] = ((true_robot_pose[2] + np.pi) % (2 * np.pi)) - np.pi
        
        # Generate synthetic observations
        sensor_frame = simulate_range_bearing_observations(
            true_robot_pose, true_landmarks, 
            max_range=15.0, timestamp=odom_action.timestamp
        )
        
        # Process with EKF SLAM
        slam.process_action_observation(odom_action, sensor_frame)
        
        # Store results
        estimated_pose = slam.get_current_robot_pose()
        estimated_poses.append(estimated_pose.copy())
        true_poses.append(true_robot_pose.copy())
        
        # Print progress
        if i % 20 == 0:
            state_summary = slam.get_state_summary()
            print(f"Step {i:3d}: Robot at ({estimated_pose[0]:6.2f}, {estimated_pose[1]:6.2f}, "
                  f"{math.degrees(estimated_pose[2]):6.1f}°), "
                  f"Landmarks: {state_summary['num_landmarks']}")
    
    # Final results
    final_landmarks = slam.get_landmark_positions()
    landmark_ids = slam.get_landmark_ids_map()
    state_summary = slam.get_state_summary()
    
    print(f"\nSimulation completed!")
    print(f"Final robot pose: ({estimated_pose[0]:.2f}, {estimated_pose[1]:.2f}, "
          f"{math.degrees(estimated_pose[2]):.1f}°)")
    print(f"Total landmarks mapped: {len(final_landmarks)}")
    print(f"True landmarks: {len(true_landmarks)}")
    
    # Compute final errors
    if len(estimated_poses) > 0:
        final_pose_error = np.linalg.norm(estimated_poses[-1][:2] - true_poses[-1][:2])
        print(f"Final position error: {final_pose_error:.3f} m")
    
    # Plot results
    plot_results(
        true_poses, estimated_poses, 
        true_landmarks, final_landmarks,
        landmark_ids
    )
    
    return slam, estimated_poses, final_landmarks


def plot_results(true_poses: List[np.ndarray], 
                estimated_poses: List[np.ndarray],
                true_landmarks: List[Tuple[float, float]], 
                estimated_landmarks: List[np.ndarray],
                landmark_ids: dict):
    """Plot the SLAM results."""
    
    plt.figure(figsize=(12, 10))
    
    # Convert to arrays for plotting
    true_poses_array = np.array(true_poses)
    est_poses_array = np.array(estimated_poses) 
    true_lm_array = np.array(true_landmarks)
    
    # Plot robot trajectories
    plt.subplot(2, 2, 1)
    plt.plot(true_poses_array[:, 0], true_poses_array[:, 1], 'b-', 
             linewidth=2, label='True trajectory')
    plt.plot(est_poses_array[:, 0], est_poses_array[:, 1], 'r--', 
             linewidth=2, label='Estimated trajectory')
    plt.scatter(true_lm_array[:, 0], true_lm_array[:, 1], 
                c='green', marker='s', s=100, label='True landmarks')
    
    if estimated_landmarks:
        est_lm_array = np.array(estimated_landmarks)
        plt.scatter(est_lm_array[:, 0], est_lm_array[:, 1], 
                    c='red', marker='o', s=80, label='Estimated landmarks')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Robot Trajectory and Landmarks')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot position errors over time
    plt.subplot(2, 2, 2)
    if len(true_poses) == len(estimated_poses):
        pos_errors = [np.linalg.norm(est[:2] - true[:2]) 
                     for est, true in zip(estimated_poses, true_poses)]
        time_steps = range(len(pos_errors))
        plt.plot(time_steps, pos_errors, 'b-', linewidth=2)
        plt.xlabel('Time step')
        plt.ylabel('Position error (m)')
        plt.title('Position Error Over Time')
        plt.grid(True)
    
    # Plot orientation errors
    plt.subplot(2, 2, 3) 
    if len(true_poses) == len(estimated_poses):
        angle_errors = [abs(((est[2] - true[2] + np.pi) % (2 * np.pi)) - np.pi) 
                       for est, true in zip(estimated_poses, true_poses)]
        angle_errors_deg = [math.degrees(err) for err in angle_errors]
        plt.plot(time_steps, angle_errors_deg, 'r-', linewidth=2)
        plt.xlabel('Time step')
        plt.ylabel('Orientation error (degrees)')
        plt.title('Orientation Error Over Time')
        plt.grid(True)
    
    # Plot landmark errors  
    plt.subplot(2, 2, 4)
    if estimated_landmarks and len(estimated_landmarks) > 0:
        lm_errors = []
        matched_pairs = []
        
        for lm_id, est_idx in landmark_ids.items():
            if lm_id < len(true_landmarks) and est_idx < len(estimated_landmarks):
                true_lm = np.array(true_landmarks[lm_id])
                est_lm = estimated_landmarks[est_idx]
                error = np.linalg.norm(est_lm - true_lm)
                lm_errors.append(error)
                matched_pairs.append(lm_id)
        
        if lm_errors:
            plt.bar(range(len(lm_errors)), lm_errors)
            plt.xlabel('Landmark ID')
            plt.ylabel('Position error (m)')
            plt.title('Landmark Position Errors')
            plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the example."""
    np.random.seed(42)  # For reproducible results
    
    try:
        slam_filter, poses, landmarks = run_ekf_slam_simulation()
        
        print("\nExample completed successfully!")
        print(f"Final state vector length: {len(slam_filter.state_vector)}")
        print(f"Final covariance matrix size: {slam_filter.covariance_matrix.shape}")
        
        # Show some statistics
        data_assoc = slam_filter.get_last_data_association()
        print(f"Last step associations: {len(data_assoc.associations)}")
        print(f"Total new landmarks added: {len(data_assoc.newly_inserted_landmarks)}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 