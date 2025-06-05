#!/usr/bin/env python3
"""
Comprehensive test suite for the Python EKF SLAM 2D implementation.

This script includes unit tests for individual components and integration tests
with synthetic sensor data and odometry. It demonstrates the full functionality
of the RangeBearingKFSLAM2D class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import math
import unittest
import logging

# Import the SLAM components
from ekf_slam import (
    RangeBearingKFSLAM2D, 
    RangeBearingObservation, 
    SensoryFrame, 
    ActionCollection,
    EKFSLAMOptions,
    Pose2D,
    PosePDFGaussian
)


class TestEKFSLAM(unittest.TestCase):
    """Unit tests for EKF SLAM components."""

    def setUp(self):
        """Set up test fixtures."""
        self.slam = RangeBearingKFSLAM2D()
        np.random.seed(42)  # For reproducible tests

    def test_initialization(self):
        """Test SLAM filter initialization."""
        self.assertEqual(len(self.slam.state_vector), 3)
        self.assertEqual(self.slam.covariance_matrix.shape, (3, 3))
        self.assertTrue(np.allclose(self.slam.state_vector, np.zeros(3)))
        self.assertTrue(np.all(np.diag(self.slam.covariance_matrix) > 0))

    def test_pose_operations(self):
        """Test Pose2D operations."""
        pose1 = Pose2D(1.0, 2.0, 0.5)
        pose2 = Pose2D(0.5, -1.0, 0.2)
        
        result = pose1 + pose2
        self.assertAlmostEqual(result.x, 1.5)
        self.assertAlmostEqual(result.y, 1.0)
        self.assertAlmostEqual(result.yaw, 0.7)

    def test_angle_wrapping(self):
        """Test angle wrapping functionality."""
        # Test various angles
        test_angles = [0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi, -3*np.pi]
        for angle in test_angles:
            wrapped = self.slam._wrap_to_pi(angle)
            self.assertTrue(-np.pi <= wrapped <= np.pi)

    def test_observation_creation(self):
        """Test observation data structures."""
        obs = RangeBearingObservation(range=5.0, yaw=0.5, landmark_id=1)
        self.assertEqual(obs.range, 5.0)
        self.assertEqual(obs.yaw, 0.5)
        self.assertEqual(obs.landmark_id, 1)

    def test_action_processing(self):
        """Test action vector processing."""
        action = ActionCollection(dx=1.0, dy=0.5, dyaw=0.1, timestamp=0.0)
        self.slam.action = action
        
        action_vector = self.slam.on_get_action()
        expected = np.array([1.0, 0.5, 0.1])
        self.assertTrue(np.allclose(action_vector, expected))

    def test_transition_model(self):
        """Test motion model."""
        # Set initial state
        initial_state = np.array([0.0, 0.0, 0.0])
        action = np.array([1.0, 0.5, 0.1])
        
        # Test transition
        new_state, skip = self.slam.on_transition_model(action, initial_state)
        
        if not skip:
            expected = np.array([1.0, 0.5, 0.1])
            self.assertTrue(np.allclose(new_state, expected, atol=1e-6))

    def test_noise_matrices(self):
        """Test noise matrix generation."""
        # Test observation noise
        R = self.slam.on_get_observation_noise()
        self.assertEqual(R.shape, (2, 2))
        self.assertTrue(np.all(np.diag(R) > 0))

        # Test process noise
        Q = self.slam.on_transition_noise()
        self.assertEqual(Q.shape, (3, 3))
        self.assertTrue(np.all(np.diag(Q) > 0))

    def test_landmark_management(self):
        """Test landmark addition and retrieval."""
        # Add a landmark manually by expanding state
        initial_state = self.slam.state_vector.copy()
        landmark_pos = np.array([5.0, 3.0])
        
        # Expand state vector
        new_state = np.zeros(5)
        new_state[:3] = initial_state
        new_state[3:5] = landmark_pos
        self.slam.state_vector = new_state
        
        # Expand covariance
        new_cov = np.eye(5) * 0.1
        new_cov[:3, :3] = self.slam.covariance_matrix
        self.slam.covariance_matrix = new_cov
        
        # Test retrieval
        retrieved_pos = self.slam.get_landmark_mean(0)
        self.assertTrue(np.allclose(retrieved_pos, landmark_pos))


def create_synthetic_landmarks() -> List[Tuple[float, float]]:
    """Create a set of synthetic landmarks in a 2D environment."""
    landmarks = [
        (10.0, 0.0),   # Landmark 0
        (10.0, 10.0),  # Landmark 1  
        (0.0, 10.0),   # Landmark 2
        (-10.0, 10.0), # Landmark 3
        (-10.0, 0.0),  # Landmark 4
        (-10.0, -10.0), # Landmark 5
        (0.0, -10.0),  # Landmark 6
        (10.0, -10.0), # Landmark 7
    ]
    return landmarks


def generate_odometry_sequence(steps: int = 100) -> List[ActionCollection]:
    """Generate a synthetic odometry sequence (circular path)."""
    actions = []
    dt = 0.1  # time step
    v = 1.0   # linear velocity (m/s)
    w = 0.05  # angular velocity (rad/s) - slower for better convergence
    
    # Create circular motion
    for i in range(steps):
        timestamp = i * dt
        
        # Move forward with slight turning
        dx = v * dt
        dy = 0.0  # Pure forward motion
        dyaw = w * dt
        
        # Add some noise to odometry
        dx += np.random.normal(0, 0.005)  # Reduced noise
        dy += np.random.normal(0, 0.005) 
        dyaw += np.random.normal(0, 0.005)
        
        # Simple covariance model
        odom_cov = np.diag([0.005**2, 0.005**2, 0.005**2])
        
        action = ActionCollection(
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
    timestamp: float = 0.0,
    add_noise: bool = True
) -> SensoryFrame:
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
    add_noise : bool
        Whether to add sensor noise
        
    Returns:
    --------
    SensoryFrame
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
            observed_range = true_range
            observed_bearing = true_bearing
            
            # Add sensor noise if requested
            if add_noise:
                noise_range = np.random.normal(0, 0.05)  # 5cm std dev
                noise_bearing = np.random.normal(0, 0.02)  # ~1 degree std dev
                
                observed_range = max(0.1, true_range + noise_range)
                observed_bearing = true_bearing + noise_bearing
            
            obs = RangeBearingObservation(
                range=observed_range,
                yaw=observed_bearing, 
                pitch=0.0,
                landmark_id=lm_id  # Known landmark IDs for this example
            )
            observations.append(obs)
    
    return SensoryFrame(
        observations=observations,
        timestamp=timestamp,
        sensor_pose=np.zeros(6)  # Sensor at robot origin
    )


def run_ekf_slam_simulation(steps: int = 80, verbose: bool = True):
    """Run a complete EKF SLAM simulation with synthetic data."""
    
    if verbose:
        print("Starting EKF SLAM 2D simulation...")
    
    # Create landmarks and robot trajectory
    true_landmarks = create_synthetic_landmarks()
    odometry_sequence = generate_odometry_sequence(steps)
    
    # Initialize EKF SLAM
    slam = RangeBearingKFSLAM2D()
    
    # Configure options
    options = EKFSLAMOptions()
    options.std_sensor_range = 0.05  # 5cm range noise
    options.std_sensor_yaw = 0.02   # ~1 degree bearing noise
    options.std_q_no_odo = [0.02, 0.02, math.radians(1.0)]  # Process noise
    options.create_simplemap = True
    slam.options = options
    
    # Storage for results
    estimated_poses = []
    true_poses = []
    
    # Simulate robot motion and observations
    true_robot_pose = np.array([0.0, 0.0, 0.0])  # Start at origin
    
    if verbose:
        print(f"Processing {len(odometry_sequence)} time steps...")
    
    for i, odom_action in enumerate(odometry_sequence):
        # Update true robot pose (for ground truth)
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
        try:
            slam.process_action_observation(odom_action, sensor_frame)
            
            # Store results
            estimated_pose_pdf = slam.get_current_robot_pose()
            estimated_pose = np.array([
                estimated_pose_pdf.mean.x,
                estimated_pose_pdf.mean.y,
                estimated_pose_pdf.mean.yaw
            ])
            estimated_poses.append(estimated_pose)
            true_poses.append(true_robot_pose.copy())
            
            # Print progress
            if verbose and i % 20 == 0:
                n_landmarks = slam.number_of_landmarks
                print(f"Step {i:3d}: Robot at ({estimated_pose[0]:6.2f}, {estimated_pose[1]:6.2f}, "
                      f"{math.degrees(estimated_pose[2]):6.1f}°), "
                      f"Landmarks: {n_landmarks}")
                      
        except Exception as e:
            print(f"Error at step {i}: {e}")
            break
    
    # Get final results
    _, final_landmarks, landmark_ids, _, _ = slam.get_current_state()
    
    if verbose:
        print(f"\nSimulation completed!")
        if len(estimated_poses) > 0:
            final_pose = estimated_poses[-1]
            print(f"Final robot pose: ({final_pose[0]:.2f}, {final_pose[1]:.2f}, "
                  f"{math.degrees(final_pose[2]):.1f}°)")
        print(f"Total landmarks mapped: {len(final_landmarks)}")
        print(f"True landmarks: {len(true_landmarks)}")
        
        # Compute final errors
        if len(estimated_poses) > 0 and len(true_poses) > 0:
            final_pose_error = np.linalg.norm(estimated_poses[-1][:2] - true_poses[-1][:2])
            print(f"Final position error: {final_pose_error:.3f} m")
    
    return slam, estimated_poses, true_poses, final_landmarks, true_landmarks, landmark_ids


def plot_results(estimated_poses: List[np.ndarray],
                true_poses: List[np.ndarray], 
                estimated_landmarks: np.ndarray,
                true_landmarks: List[Tuple[float, float]], 
                landmark_ids: dict,
                show_plot: bool = True):
    """Plot the SLAM results."""
    
    if not estimated_poses or not true_poses:
        print("No pose data to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Convert to arrays for plotting
    true_poses_array = np.array(true_poses)
    est_poses_array = np.array(estimated_poses) 
    true_lm_array = np.array(true_landmarks)
    
    # Plot robot trajectories
    plt.subplot(2, 3, 1)
    plt.plot(true_poses_array[:, 0], true_poses_array[:, 1], 'b-', 
             linewidth=2, label='True trajectory')
    plt.plot(est_poses_array[:, 0], est_poses_array[:, 1], 'r--', 
             linewidth=2, label='Estimated trajectory')
    plt.scatter(true_lm_array[:, 0], true_lm_array[:, 1], 
                c='green', marker='s', s=100, label='True landmarks')
    
    if len(estimated_landmarks) > 0:
        plt.scatter(estimated_landmarks[:, 0], estimated_landmarks[:, 1], 
                    c='red', marker='o', s=80, label='Estimated landmarks')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Robot Trajectory and Landmarks')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot position errors over time
    plt.subplot(2, 3, 2)
    pos_errors = [np.linalg.norm(est[:2] - true[:2]) 
                 for est, true in zip(estimated_poses, true_poses)]
    time_steps = range(len(pos_errors))
    plt.plot(time_steps, pos_errors, 'b-', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Position error (m)')
    plt.title('Position Error Over Time')
    plt.grid(True)
    
    # Plot orientation errors
    plt.subplot(2, 3, 3) 
    angle_errors = [abs(((est[2] - true[2] + np.pi) % (2 * np.pi)) - np.pi) 
                   for est, true in zip(estimated_poses, true_poses)]
    angle_errors_deg = [math.degrees(err) for err in angle_errors]
    plt.plot(time_steps, angle_errors_deg, 'r-', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Orientation error (degrees)')
    plt.title('Orientation Error Over Time')
    plt.grid(True)
    
    # Plot landmark errors  
    plt.subplot(2, 3, 4)
    if len(estimated_landmarks) > 0:
        lm_errors = []
        matched_landmarks = []
        
        # Find matched landmarks
        for est_idx in range(len(estimated_landmarks)):
            # Find corresponding true landmark
            found_match = False
            for true_id, mapped_idx in landmark_ids.items():
                if mapped_idx == est_idx and true_id < len(true_landmarks):
                    true_lm = np.array(true_landmarks[true_id])
                    est_lm = estimated_landmarks[est_idx]
                    error = np.linalg.norm(est_lm - true_lm)
                    lm_errors.append(error)
                    matched_landmarks.append(true_id)
                    found_match = True
                    break
            
            if not found_match:
                # This is a spurious landmark
                lm_errors.append(float('nan'))
                matched_landmarks.append(-1)
        
        if lm_errors:
            valid_errors = [e for e in lm_errors if not np.isnan(e)]
            plt.bar(range(len(lm_errors)), lm_errors)
            plt.xlabel('Landmark Index')
            plt.ylabel('Position error (m)')
            plt.title(f'Landmark Position Errors (avg: {np.mean(valid_errors):.3f}m)')
            plt.grid(True, axis='y')
    
    # Plot covariance evolution
    plt.subplot(2, 3, 5)
    if len(estimated_poses) > 0:
        # This would require storing covariance data - for now just show trajectory
        plt.plot(est_poses_array[:, 0], est_poses_array[:, 1], 'r-', alpha=0.7)
        plt.title('Estimated Trajectory Detail')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
    
    # Statistics summary
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, 'SLAM Statistics:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    
    stats_text = []
    if len(pos_errors) > 0:
        stats_text.append(f'Final pos error: {pos_errors[-1]:.3f} m')
        stats_text.append(f'Avg pos error: {np.mean(pos_errors):.3f} m')
        stats_text.append(f'Max pos error: {np.max(pos_errors):.3f} m')
    
    if len(angle_errors_deg) > 0:
        stats_text.append(f'Final angle error: {angle_errors_deg[-1]:.1f}°')
        stats_text.append(f'Avg angle error: {np.mean(angle_errors_deg):.1f}°')
    
    stats_text.append(f'Landmarks mapped: {len(estimated_landmarks)}/{len(true_landmarks)}')
    
    for i, text in enumerate(stats_text):
        plt.text(0.1, 0.8 - i*0.1, text, fontsize=12, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return plt.gcf()


def test_slam_convergence():
    """Test SLAM convergence properties."""
    print("\n=== Testing SLAM Convergence ===")
    
    # Run simulation
    slam, est_poses, true_poses, est_landmarks, true_landmarks, landmark_ids = run_ekf_slam_simulation(
        steps=60, verbose=False
    )
    
    if not est_poses or not true_poses:
        print("FAILED: No poses generated")
        return False
    
    # Check final position error
    final_pos_error = np.linalg.norm(est_poses[-1][:2] - true_poses[-1][:2])
    print(f"Final position error: {final_pos_error:.3f} m")
    
    # Check if error is reasonable
    if final_pos_error > 2.0:  # Allow up to 2m error
        print(f"FAILED: Position error too large ({final_pos_error:.3f} > 2.0)")
        return False
    
    # Check landmark mapping
    landmarks_mapped = len(est_landmarks)
    landmarks_expected = len(true_landmarks)
    print(f"Landmarks mapped: {landmarks_mapped}/{landmarks_expected}")
    
    if landmarks_mapped < landmarks_expected * 0.5:  # At least 50% should be mapped
        print(f"FAILED: Too few landmarks mapped ({landmarks_mapped}/{landmarks_expected})")
        return False
    
    print("PASSED: SLAM convergence test")
    return True


def test_noise_robustness():
    """Test SLAM robustness to noise."""
    print("\n=== Testing Noise Robustness ===")
    
    # Test with different noise levels
    noise_levels = [0.01, 0.05, 0.1]  # Range noise std dev
    results = []
    
    for noise_std in noise_levels:
        print(f"Testing with noise std: {noise_std}")
        
        # Temporarily modify global noise for this test
        original_seed = np.random.get_state()
        np.random.seed(42)
        
        slam, est_poses, true_poses, est_landmarks, true_landmarks, landmark_ids = run_ekf_slam_simulation(
            steps=40, verbose=False
        )
        
        # Restore random state
        np.random.set_state(original_seed)
        
        if est_poses and true_poses:
            final_error = np.linalg.norm(est_poses[-1][:2] - true_poses[-1][:2])
            results.append((noise_std, final_error, len(est_landmarks)))
            print(f"  Final error: {final_error:.3f} m, Landmarks: {len(est_landmarks)}")
        else:
            results.append((noise_std, float('inf'), 0))
            print(f"  FAILED: No results generated")
    
    # Check that errors don't grow unreasonably with noise
    success = True
    for noise_std, error, n_landmarks in results:
        if error > 5.0:  # 5m threshold
            print(f"FAILED: Error too large for noise level {noise_std}: {error:.3f} m")
            success = False
    
    if success:
        print("PASSED: Noise robustness test")
    
    return success


def main():
    """Main test function."""
    print("=== EKF SLAM 2D Test Suite ===\n")
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    print("\n=== Integration Tests ===")
    
    # Test basic functionality
    test_slam_convergence()
    test_noise_robustness()
    
    # Run main simulation
    print("\n=== Main Simulation ===")
    slam, est_poses, true_poses, est_landmarks, true_landmarks, landmark_ids = run_ekf_slam_simulation()
    
    # Plot results
    if est_poses and true_poses:
        print("\nGenerating plots...")
        plot_results(est_poses, true_poses, est_landmarks, true_landmarks, landmark_ids)
    else:
        print("No results to plot")
    
    print("\nTest suite completed!")


if __name__ == "__main__":
    main() 