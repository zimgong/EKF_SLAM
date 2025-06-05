"""
EKF SLAM 2D implementation for range-bearing sensors.
Python translation of MRPT's CRangeBearingKFSLAM2D class.

This module provides a complete EKF-based SLAM implementation for 2D environments
using range-bearing sensor observations and odometry data.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import math

from kalman_filter import KalmanFilterCapable, KFMethod


@dataclass
class RangeBearingObservation:
    """Single range-bearing observation."""

    range: float
    yaw: float
    pitch: float = 0.0
    landmark_id: int = -1  # -1 means unknown/new landmark


@dataclass
class ActionCollection:
    """Odometry action/motion command."""

    dx: float  # translation in x
    dy: float  # translation in y
    dyaw: float  # rotation change
    timestamp: float
    covariance: np.ndarray = None  # 3x3 covariance matrix


@dataclass
class SensoryFrame:
    """Collection of sensor observations at a timestamp."""

    observations: List[RangeBearingObservation]
    timestamp: float
    sensor_pose: np.ndarray = None  # 3D pose [x, y, z, roll, pitch, yaw]


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float

    def __add__(self, other: 'Pose2D') -> 'Pose2D':
        new_x = self.x + other.x
        new_y = self.y + other.y
        new_yaw = self._wrap_to_pi(self.yaw + other.yaw)
        return Pose2D(new_x, new_y, new_yaw)

    def _wrap_to_pi(self, angle: float) -> float:
        """Wrap angle to [-π, π] range."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi


@dataclass
class PosePDFGaussian:
    mean: Pose2D
    cov: np.ndarray

@dataclass
class EKFSLAMOptions:
    """Configuration options for EKF SLAM 2D."""

    # Process noise when no odometry is available
    std_q_no_odo: List[float] = None  # [std_x, std_y, std_yaw] in meters and radians

    # Sensor noise parameters
    std_sensor_range: float = 0.1  # meters
    std_sensor_yaw: float = 0.1  # radians

    # Visualization parameters
    quantiles_3d_representation: float = 3.0
    create_simplemap: bool = False

    # Data association parameters
    data_assoc_method: str = (
        "nearest_neighbor"  # "nearest_neighbor", "joint_compatibility"
    )
    data_assoc_metric: str = "mahalanobis"  # "mahalanobis", "euclidean"
    data_assoc_chi2_threshold: float = 0.99
    data_assoc_ml_threshold: float = 0.0

    def __post_init__(self):
        if self.std_q_no_odo is None:
            self.std_q_no_odo = [0.1, 0.1, math.radians(4.0)]


class DataAssociationInfo:
    """Information about the last data association step."""

    def __init__(self):
        self.prediction_means: List[np.ndarray] = []
        self.prediction_covariances: np.ndarray = np.array([])
        self.prediction_IDs: List[int] = []
        self.associations: Dict[int, int] = {}  # obs_idx -> landmark_idx
        self.newly_inserted_landmarks: Dict[int, int] = {}  # obs_idx -> landmark_idx

    def clear(self):
        self.prediction_means.clear()
        self.prediction_covariances = np.array([])
        self.prediction_IDs.clear()
        self.associations.clear()
        self.newly_inserted_landmarks.clear()


def get_inverse_map(map: Dict[int, int]) -> Dict[int, int]:
    return {v: k for k, v in map.items()}


class RangeBearingKFSLAM2D(KalmanFilterCapable):
    """
    EKF-based SLAM implementation for 2D environments with range-bearing sensors.

    This class implements Simultaneous Localization and Mapping using an Extended
    Kalman Filter for 2D robot poses and 2D landmark positions, with range-bearing
    sensor observations.

    State vector format: [robot_x, robot_y, robot_yaw, lm1_x, lm1_y, lm2_x, lm2_y, ...]
    """

    def __init__(self):
        super().__init__(
            vehicle_size=3, observation_size=2, feature_size=2, action_size=3
        )

        # EKF SLAM specific options
        self.options = EKFSLAMOptions()

        self.action: Optional[ActionCollection] = None
        self.SF: Optional[SensoryFrame] = None
        # The mapping between landmark IDs and indexes in the Pkk cov. matrix:
        self.IDs: Dict[int, int] = {}
        # The sequence of all the observations and the robot path (kept for
        # debugging, statistics,etc)
        self.SFs: List[Tuple[Pose2D, SensoryFrame]] = []

        # Data association info
        self.last_data_association = DataAssociationInfo()

        self.reset()

    def reset(self):
        self.action = None
        self.SF = None
        self.IDs = {}
        self.SFs = []

        # Init KF state
        self.state_vector = np.zeros(3)  # State: 3D pose (x, y, phi)
        # Initial cov:
        self.covariance_matrix = np.eye(3) * 0.1  # Small initial uncertainty

    def get_current_robot_pose(self) -> PosePDFGaussian:
        pose_mean = Pose2D(self.state_vector[0], self.state_vector[1], self.state_vector[2])
        return PosePDFGaussian(
            mean=pose_mean,
            cov=self.covariance_matrix[:3, :3].copy(),
        )

    def get_current_state(self) -> Tuple[Pose2D, np.ndarray, Dict[int, int], np.ndarray, np.ndarray]:
        # Set pose mean and cov:
        robot_pose = self.get_current_robot_pose()

        # Landmarks: 
        n_LMs = (len(self.state_vector) - 3) // 2
        landmarks_positions = np.zeros((n_LMs, 2))
        for i in range(n_LMs):
            landmarks_positions[i] = self.state_vector[3 + i * 2 : 3 + i * 2 + 2]

        # IDs:
        landmark_IDs = get_inverse_map(self.IDs)

        # Full state:
        full_state = self.state_vector.copy()

        # Full cov:
        full_covariance = self.covariance_matrix.copy()

        return robot_pose, landmarks_positions, landmark_IDs, full_state, full_covariance

    def process_action_observation(self, action: ActionCollection, SF: SensoryFrame):
        self.action = action
        self.SF = SF

        # Here's the meat!: Call the main method for the KF algorithm, which will
        # call all the callback methods as required:
        self.run_one_kalman_iteration()

        # Add to SFs sequence
        if self.options.create_simplemap:
            p = self.get_current_robot_pose()
            self.SFs.append((p.mean, SF))

    def on_get_action(self) -> np.ndarray:
        """
        Return the current action vector u

        Returns:
            u: The action vector which will be passed to on_transition_model
        """
        if self.action:
            return np.array([self.action.dx, self.action.dy, self.action.dyaw])
        else:
            return np.zeros(3)

    def on_transition_model(self, u: np.ndarray, xv: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Implement the prediction model of the Kalman filter.
        """
        # Don't update the vehicle pose & its covariance until we have some
        # landmarks in the map, 
        # otherwise, we are imposing a lower bound to the best uncertainty from now 
        # on:
        if len(self.state_vector) == self.vehicle_size:
            return xv, True  # Skip prediction
        
        robot_pose = Pose2D(xv[0], xv[1], xv[2])
        odo_increment = Pose2D(u[0], u[1], u[2])

        robot_pose = robot_pose + odo_increment

        new_state = np.array([robot_pose.x, robot_pose.y, robot_pose.yaw])

        return new_state, False

    def on_transition_jacobian(self) -> np.ndarray:
        """Compute the Jacobian of the motion model."""
        if self.action is None:
            return np.eye(3)

        # Get current robot orientation
        yaw = self.state_vector[2]

        # Odometry increment
        dx = self.action.dx
        dy = self.action.dy

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Jacobian matrix
        F = np.eye(3)
        F[0, 2] = -dx * sin_yaw - dy * cos_yaw  # ∂x/∂yaw
        F[1, 2] = dx * cos_yaw - dy * sin_yaw  # ∂y/∂yaw

        return F

    def on_transition_noise(self) -> np.ndarray:
        """Return the process noise covariance matrix."""
        if self.action is None or self.action.covariance is None:
            # Use default noise model when no odometry covariance available
            Q = np.diag(
                [
                    self.options.std_q_no_odo[0] ** 2,  # x variance
                    self.options.std_q_no_odo[1] ** 2,  # y variance
                    self.options.std_q_no_odo[2] ** 2,  # yaw variance
                ]
            )
        else:
            # Use provided covariance, but rotate to global frame
            Q = self.action.covariance.copy()

            # Rotate covariance by current robot orientation
            yaw = self.state_vector[2]
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            # Rotation matrix for [x, y, yaw] -> only rotate x,y components
            R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

            Q = R @ Q @ R.T

        return Q

    def on_get_observation_noise(self) -> np.ndarray:
        """Return observation noise covariance matrix."""
        return np.diag(
            [
                self.options.std_sensor_range**2,  # range variance
                self.options.std_sensor_yaw**2,  # bearing variance
            ]
        )

    def on_observation_model(self, landmark_indices: List[int]) -> List[np.ndarray]:
        """
        Predict range-bearing observations to landmarks.

        h(x) = [range, bearing] for each landmark
        """
        if self.SF is None:
            return []

        # Get robot pose and sensor pose
        robot_pose = self.state_vector[:3]  # [x, y, yaw]
        sensor_pose_rel = self.SF.sensor_pose

        if sensor_pose_rel is None:
            sensor_pose_rel = np.zeros(6)  # Assume sensor at robot origin

        # Compute absolute sensor pose
        sensor_pose_abs = self._compose_poses_2d(robot_pose, sensor_pose_rel[:3])

        predictions = []
        for lm_idx in landmark_indices:

            # Get landmark position
            lm_pos = self.get_landmark_mean(lm_idx)  # [x, y]

            # Relative position from sensor to landmark
            dx = lm_pos[0] - sensor_pose_abs[0]
            dy = lm_pos[1] - sensor_pose_abs[1]

            # Compute range and bearing
            range_pred = np.sqrt(dx**2 + dy**2)
            bearing_pred = self._wrap_to_pi(np.arctan2(dy, dx) - sensor_pose_abs[2])

            predictions.append(np.array([range_pred, bearing_pred]))

        return predictions

    def on_observation_jacobians(
        self, landmark_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute observation Jacobians for range-bearing observations.

        Returns Hx (∂h/∂robot_pose) and Hy (∂h/∂landmark_pos)
        """
        if self.SF is None:
            Hx = np.zeros((2, 3))
            Hy = np.zeros((2, 2))
            return Hx, Hy

        # Get poses
        robot_pose = self.state_vector[:3]
        sensor_pose_rel = self.SF.sensor_pose
        if sensor_pose_rel is None:
            sensor_pose_rel = np.zeros(6)

        # Sensor pose in global frame
        sensor_pose_abs = self._compose_poses_2d(robot_pose, sensor_pose_rel[:3])
        sx, sy, syaw = sensor_pose_abs[0], sensor_pose_abs[1], sensor_pose_abs[2]

        # Landmark position
        lm_pos = self.get_landmark_mean(landmark_idx)
        lx, ly = lm_pos[0], lm_pos[1]

        # Relative position
        dx = lx - sx
        dy = ly - sy

        # Distance
        r = np.sqrt(dx**2 + dy**2)
        if r < 1e-6:
            r = 1e-6  # Avoid division by zero

        r_sq = r**2

        # Sensor pose relative to robot
        sx_rel = sensor_pose_rel[0]
        sy_rel = sensor_pose_rel[1]
        syaw_rel = sensor_pose_rel[2]

        # Robot orientation
        robot_yaw = robot_pose[2]
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        # Jacobian w.r.t. robot pose (3x1 -> 2x3)
        Hx = np.zeros((2, 3))

        # ∂range/∂robot_pose
        Hx[0, 0] = -dx / r  # ∂range/∂robot_x
        Hx[0, 1] = -dy / r  # ∂range/∂robot_y
        # ∂range/∂robot_yaw (considering sensor offset)
        Hx[0, 2] = (
            dx * (sy_rel * cos_yaw + sx_rel * sin_yaw)
            + dy * (sx_rel * cos_yaw - sy_rel * sin_yaw)
        ) / r

        # ∂bearing/∂robot_pose
        Hx[1, 0] = dy / r_sq  # ∂bearing/∂robot_x
        Hx[1, 1] = -dx / r_sq  # ∂bearing/∂robot_y
        # ∂bearing/∂robot_yaw
        Hx[1, 2] = (
            -dy * (sy_rel * cos_yaw + sx_rel * sin_yaw)
            + dx * (sx_rel * cos_yaw - sy_rel * sin_yaw)
        ) / r_sq - 1.0

        # Jacobian w.r.t. landmark position (2x1 -> 2x2)
        Hy = np.zeros((2, 2))

        # ∂range/∂landmark_pos
        Hy[0, 0] = dx / r  # ∂range/∂landmark_x
        Hy[0, 1] = dy / r  # ∂range/∂landmark_y

        # ∂bearing/∂landmark_pos
        Hy[1, 0] = -dy / r_sq  # ∂bearing/∂landmark_x
        Hy[1, 1] = dx / r_sq  # ∂bearing/∂landmark_y

        return Hx, Hy

    def on_get_observations_and_data_association(
        self,
        all_predictions: List[np.ndarray],
        innovation_cov: np.ndarray,
        landmark_indices: List[int],
        obs_noise: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        This is called between the KF prediction step and the update step, and the
        application must return the observations and, when applicable, the data
        association between these observations and the current map.
        """
        if self.SF is None:
            return [], []

        observations = []
        data_association = []

        # Convert observations to numpy arrays
        for obs in self.SF.observations:
            observations.append(np.array([obs.range, obs.yaw]))

        # Simple data association based on landmark IDs
        for i, obs in enumerate(self.SF.observations):
            if obs.landmark_id >= 0 and obs.landmark_id in self.IDs:
                # Known landmark
                data_association.append(self.IDs[obs.landmark_id])
            else:
                # New landmark
                data_association.append(-1)

        # Store data association info
        self.last_data_association.clear()
        self.last_data_association.prediction_means = all_predictions
        self.last_data_association.prediction_covariances = innovation_cov
        self.last_data_association.prediction_IDs = landmark_indices

        for i, assoc in enumerate(data_association):
            if assoc >= 0:
                self.last_data_association.associations[i] = assoc

        return observations, data_association

    def on_inverse_observation_model(
        self, observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Compute inverse observation model for new landmark initialization.

        Given: robot pose, sensor pose, and observation [range, bearing]
        Compute: landmark position and Jacobians
        """
        if self.SF is None:
            raise ValueError("No current sensor frame available")

        # Get sensor pose
        robot_pose = self.state_vector[:3]
        sensor_pose_rel = self.SF.sensor_pose
        if sensor_pose_rel is None:
            sensor_pose_rel = np.zeros(6)

        sensor_pose_abs = self._compose_poses_2d(robot_pose, sensor_pose_rel[:3])

        # Observation
        range_obs, bearing_obs = observation[0], observation[1]

        # Compute landmark position in global frame
        sensor_x, sensor_y, sensor_yaw = sensor_pose_abs
        landmark_x = sensor_x + range_obs * np.cos(sensor_yaw + bearing_obs)
        landmark_y = sensor_y + range_obs * np.sin(sensor_yaw + bearing_obs)

        landmark_pos = np.array([landmark_x, landmark_y])

        # Jacobian w.r.t. robot pose (2x3)
        cos_bearing = np.cos(sensor_yaw + bearing_obs)
        sin_bearing = np.sin(sensor_yaw + bearing_obs)

        # Sensor pose derivatives w.r.t. robot pose
        sx_rel, sy_rel = sensor_pose_rel[0], sensor_pose_rel[1]
        robot_yaw = robot_pose[2]
        cos_robot = np.cos(robot_yaw)
        sin_robot = np.sin(robot_yaw)

        jac_vehicle = np.zeros((2, 3))
        jac_vehicle[0, 0] = 1.0  # ∂lm_x/∂robot_x
        jac_vehicle[1, 1] = 1.0  # ∂lm_y/∂robot_y

        # ∂lm/∂robot_yaw (considering sensor offset)
        jac_vehicle[0, 2] = (
            -sx_rel * sin_robot - sy_rel * cos_robot - range_obs * sin_bearing
        )
        jac_vehicle[1, 2] = (
            sx_rel * cos_robot - sy_rel * sin_robot + range_obs * cos_bearing
        )

        # Jacobian w.r.t. observation (2x2)
        jac_obs = np.array(
            [
                [cos_bearing, -range_obs * sin_bearing],  # ∂lm_x/∂[range, bearing]
                [sin_bearing, range_obs * cos_bearing],  # ∂lm_y/∂[range, bearing]
            ]
        )

        return landmark_pos, jac_vehicle, jac_obs, True

    def on_new_landmark_added_to_map(self, obs_index: int, feature_index: int):
        """Handle addition of new landmark to the map."""
        if self.SF is None:
            return

        obs = self.SF.observations[obs_index]

        # Assign landmark ID
        if obs.landmark_id >= 0:
            # Sensor provided landmark ID
            self.IDs[obs.landmark_id] = feature_index
        else:
            # Generate new ID
            new_id = len(self.IDs)
            self.IDs[new_id] = feature_index

        # Record for statistics
        self.last_data_association.newly_inserted_landmarks[obs_index] = feature_index

    def on_normalize_state_vector(self):
        """Normalize angles in the state vector."""
        # Normalize robot yaw angle
        self.state_vector[2] = self._wrap_to_pi(self.state_vector[2])

    def on_subtract_observation_vectors(
        self, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Subtract observation vectors, handling angle wrapping for bearing."""
        result = a - b
        result[1] = self._wrap_to_pi(result[1])  # Wrap bearing difference
        return result

    def on_pre_computing_predictions(
        self, all_prediction_means: List[np.ndarray]
    ) -> List[int]:
        """
        Filter landmarks to predict based on sensor range and field of view.
        """
        if self.SF is None:
            return list(range(len(all_prediction_means)))

        # Assume some reasonable sensor limits
        max_range = 50.0  # meters
        max_bearing = np.pi  # full 360 degree FOV

        # Get current robot uncertainty for conservative prediction
        robot_cov = self.covariance_matrix[:3, :3]
        max_pos_uncertainty = 4 * np.sqrt(np.trace(robot_cov[:2, :2]))
        max_yaw_uncertainty = 4 * np.sqrt(robot_cov[2, 2])

        valid_indices = []
        for i, pred in enumerate(all_prediction_means):
            if len(pred) >= 2:
                pred_range, pred_bearing = pred[0], pred[1]

                # Conservative range check
                if (
                    pred_range
                    < max_range
                    + max_pos_uncertainty
                    + 4 * self.options.std_sensor_range
                ):
                    # Conservative bearing check
                    if (
                        abs(pred_bearing)
                        < max_bearing
                        + max_yaw_uncertainty
                        + 4 * self.options.std_sensor_yaw
                    ):
                        valid_indices.append(i)

        return valid_indices

    # Helper methods

    def _wrap_to_pi(self, angle: float) -> float:
        """Wrap angle to [-π, π] range."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def _compose_poses_2d(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """
        Compose two 2D poses: result = pose1 ⊕ pose2

        Parameters:
        -----------
        pose1, pose2 : np.ndarray
            Poses as [x, y, yaw]

        Returns:
        --------
        np.ndarray
            Composed pose [x, y, yaw]
        """
        x1, y1, yaw1 = pose1[0], pose1[1], pose1[2]
        x2, y2, yaw2 = pose2[0], pose2[1], pose2[2]

        cos_yaw1 = np.cos(yaw1)
        sin_yaw1 = np.sin(yaw1)

        result_x = x1 + x2 * cos_yaw1 - y2 * sin_yaw1
        result_y = y1 + x2 * sin_yaw1 + y2 * cos_yaw1
        result_yaw = self._wrap_to_pi(yaw1 + yaw2)

        return np.array([result_x, result_y, result_yaw])

    def load_options_from_config(self, config: Dict[str, Any]):
        """Load configuration options from a dictionary."""
        if "std_q_no_odo" in config:
            self.options.std_q_no_odo = config["std_q_no_odo"]
        if "std_sensor_range" in config:
            self.options.std_sensor_range = config["std_sensor_range"]
        if "std_sensor_yaw" in config:
            self.options.std_sensor_yaw = config["std_sensor_yaw"]
        if "data_assoc_chi2_threshold" in config:
            self.options.data_assoc_chi2_threshold = config["data_assoc_chi2_threshold"]

        # Update KF options if provided
        if "kf_method" in config:
            method_map = {
                "ekf_naive": KFMethod.EKF_NAIVE,
                "ekf_ala_davison": KFMethod.EKF_ALA_DAVISON,
                "ikf_full": KFMethod.IKF_FULL,
                "ikf": KFMethod.IKF,
            }
            if config["kf_method"] in method_map:
                self.kf_options.method = method_map[config["kf_method"]]
