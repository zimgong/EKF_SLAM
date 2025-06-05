
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple
import time
import logging
from dataclasses import dataclass
import numpy as np


class KFMethod(Enum):
    """Kalman Filter algorithm variants."""
    EKF_NAIVE = 0
    EKF_ALA_DAVISON = 1
    IKF_FULL = 2
    IKF = 3


@dataclass
class KFOptions:
    """Configuration options for the Kalman Filter algorithm."""
    
    method: KFMethod = KFMethod.EKF_NAIVE
    ikf_iterations: int = 5
    enable_profiler: bool = False
    use_analytic_transition_jacobian: bool = True
    use_analytic_observation_jacobian: bool = True
    debug_verify_analytic_jacobians: bool = False
    debug_verify_analytic_jacobians_threshold: float = 1e-2
    verbosity_level: int = logging.INFO
    
    def load_from_config(self, config_dict: dict) -> None:
        """Load options from a configuration dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def dump_to_text(self) -> str:
        """Return a string representation of all options."""
        lines = ["\n----------- [KF_options] ------------ \n"]
        lines.append(f"method                                  = {self.method.name}")
        lines.append(f"ikf_iterations                          = {self.ikf_iterations}")
        lines.append(f"enable_profiler                         = {'Y' if self.enable_profiler else 'N'}")
        lines.append(f"use_analytic_transition_jacobian        = {'Y' if self.use_analytic_transition_jacobian else 'N'}")
        lines.append(f"use_analytic_observation_jacobian       = {'Y' if self.use_analytic_observation_jacobian else 'N'}")
        lines.append(f"debug_verify_analytic_jacobians         = {'Y' if self.debug_verify_analytic_jacobians else 'N'}")
        lines.append(f"debug_verify_analytic_jacobians_threshold = {self.debug_verify_analytic_jacobians_threshold}")
        return "\n".join(lines) + "\n"


class TimeLogger:
    """Simple time profiler for performance monitoring."""
    
    def __init__(self):
        self.times = {}
        self.enabled = False
    
    def enable(self, enabled: bool = True):
        self.enabled = enabled
    
    def tic(self, name: str):
        if self.enabled:
            self.times[name] = time.time()
    
    def toc(self, name: str) -> float:
        if self.enabled and name in self.times:
            elapsed = time.time() - self.times[name]
            logging.debug(f"Timer '{name}': {elapsed:.6f} seconds")
            return elapsed
        return 0.0
    
    def get_stats(self) -> dict:
        return self.times.copy()


class KalmanFilterCapable(ABC):
    """
    Generic Kalman Filter implementation for various estimation problems.
    
    This class provides a framework for implementing Extended Kalman Filters (EKF),
    Iterative Extended Kalman Filters (IEKF), and other variants. It's particularly
    suited for SLAM applications but can be used for general state estimation.
    
    Parameters:
    -----------
    vehicle_size : int
        Dimension of the vehicle state vector
    observation_size : int
        Dimension of each observation vector
    feature_size : int
        Dimension of map features (0 if not applicable)
    action_size : int
        Dimension of action/control vectors (0 if not applicable)
    """
    
    def __init__(self, vehicle_size: int, observation_size: int, 
                 feature_size: int = 0, action_size: int = 0):
        self.vehicle_size = vehicle_size
        self.observation_size = observation_size
        self.feature_size = feature_size
        self.action_size = action_size
        
        # State vector and covariance matrix
        self.state_vector = np.zeros(vehicle_size)  # Will be resized as needed
        self.covariance_matrix = np.eye(vehicle_size)  # Will be resized as needed
        
        # Configuration and utilities
        self.kf_options = KFOptions()
        self.time_logger = TimeLogger()
        self.logger = logging.getLogger(__name__)
        
        # Internal variables for iteration
        self._user_didnt_implement_jacobian = True
        self._all_predictions = []
        self._predict_landmark_indices = []
        self._observation_jacobians_x = []
        self._observation_jacobians_y = []
        self._innovation_covariance = None
        self._observations = []
        self._kalman_gain = None
    
    @property
    def state_vector_length(self) -> int:
        """Get the length of the current state vector."""
        return len(self.state_vector)
    
    @property
    def number_of_landmarks(self) -> int:
        """Get the number of landmarks in the map."""
        if self.feature_size == 0:
            return 0
        return (len(self.state_vector) - self.vehicle_size) // self.feature_size
    
    @property
    def is_map_empty(self) -> bool:
        """Check if the map is empty (no landmarks)."""
        return self.number_of_landmarks == 0
    
    def get_landmark_mean(self, idx: int) -> np.ndarray:
        """
        Get the mean estimate of the idx-th landmark.
        
        Parameters:
        -----------
        idx : int
            Index of the landmark
            
        Returns:
        --------
        np.ndarray
            Mean estimate of the landmark
        """
        if idx >= self.number_of_landmarks:
            raise IndexError(f"Landmark index {idx} >= {self.number_of_landmarks}")
        
        start_idx = self.vehicle_size + idx * self.feature_size
        end_idx = start_idx + self.feature_size
        return self.state_vector[start_idx:end_idx].copy()
    
    def get_landmark_covariance(self, idx: int) -> np.ndarray:
        """
        Get the covariance estimate of the idx-th landmark.
        
        Parameters:
        -----------
        idx : int
            Index of the landmark
            
        Returns:
        --------
        np.ndarray
            Covariance matrix of the landmark
        """
        if idx >= self.number_of_landmarks:
            raise IndexError(f"Landmark index {idx} >= {self.number_of_landmarks}")
        
        start_idx = self.vehicle_size + idx * self.feature_size
        end_idx = start_idx + self.feature_size
        return self.covariance_matrix[start_idx:end_idx, start_idx:end_idx].copy()
    
    def set_landmark_mean(self, idx: int, mean: np.ndarray) -> None:
        """Set the mean estimate of the idx-th landmark."""
        if idx >= self.number_of_landmarks:
            raise IndexError(f"Landmark index {idx} >= {self.number_of_landmarks}")
        
        start_idx = self.vehicle_size + idx * self.feature_size
        end_idx = start_idx + self.feature_size
        self.state_vector[start_idx:end_idx] = mean
    
    def set_landmark_covariance(self, idx: int, cov: np.ndarray) -> None:
        """Set the covariance estimate of the idx-th landmark."""
        if idx >= self.number_of_landmarks:
            raise IndexError(f"Landmark index {idx} >= {self.number_of_landmarks}")
        
        start_idx = self.vehicle_size + idx * self.feature_size
        end_idx = start_idx + self.feature_size
        self.covariance_matrix[start_idx:end_idx, start_idx:end_idx] = cov
    
    # Abstract methods that must be implemented by derived classes
    
    @abstractmethod
    def on_get_action(self) -> np.ndarray:
        """
        Must return the action vector u_k.
        
        Returns:
        --------
        np.ndarray
            Action vector of size action_size
        """
        pass
    
    @abstractmethod
    def on_transition_model(self, action: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Implements the transition model: x_{k|k-1} = f(x_{k-1|k-1}, u_k)
        
        Parameters:
        -----------
        action : np.ndarray
            Action vector returned by on_get_action()
        state : np.ndarray
            Previous state estimate x_{k-1|k-1}
            
        Returns:
        --------
        Tuple[np.ndarray, bool]
            - Predicted state x_{k|k-1}
            - skip_prediction: Set to True to skip the prediction step
        """
        pass
    
    def on_transition_jacobian(self) -> np.ndarray:
        """
        Implements the transition Jacobian ∂f/∂x.
        
        Returns:
        --------
        np.ndarray
            Jacobian matrix of shape (vehicle_size, vehicle_size)
        """
        self._user_didnt_implement_jacobian = True
        return np.eye(self.vehicle_size)
    
    def on_transition_jacobian_numeric_increments(self) -> np.ndarray:
        """
        Return increments for numeric Jacobian estimation.
        
        Returns:
        --------
        np.ndarray
            Increments for each vehicle state dimension
        """
        return np.full(self.vehicle_size, 1e-6)
    
    @abstractmethod
    def on_transition_noise(self) -> np.ndarray:
        """
        Implements the transition noise covariance Q_k.
        
        Returns:
        --------
        np.ndarray
            Process noise covariance matrix
        """
        pass
    
    def on_pre_computing_predictions(self, all_prediction_means: List[np.ndarray]) -> List[int]:
        """
        Allow filtering which landmarks to predict (for efficiency).
        
        Parameters:
        -----------
        all_prediction_means : List[np.ndarray]
            Mean predictions for all landmarks
            
        Returns:
        --------
        List[int]
            Indices of landmarks to predict
        """
        return list(range(self.number_of_landmarks))
    
    @abstractmethod
    def on_get_observation_noise(self) -> np.ndarray:
        """
        Return the observation noise covariance matrix R.
        
        Returns:
        --------
        np.ndarray
            Observation noise covariance matrix
        """
        pass
    
    @abstractmethod
    def on_get_observations_and_data_association(self, 
                                                all_predictions: List[np.ndarray],
                                                innovation_cov: np.ndarray,
                                                landmark_indices: List[int],
                                                obs_noise: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Return observations and data association.
        
        Parameters:
        -----------
        all_predictions : List[np.ndarray]
            Predictions for all landmarks
        innovation_cov : np.ndarray
            Innovation covariance matrix
        landmark_indices : List[int]
            Indices of landmarks in innovation_cov
        obs_noise : np.ndarray
            Observation noise covariance
            
        Returns:
        --------
        Tuple[List[np.ndarray], List[int]]
            - observations: List of observation vectors
            - data_association: Association indices (-1 for new landmarks)
        """
        pass
    
    @abstractmethod
    def on_observation_model(self, landmark_indices: List[int]) -> List[np.ndarray]:
        """
        Implements the observation prediction h_i(x).
        
        Parameters:
        -----------
        landmark_indices : List[int]
            Indices of landmarks to predict
            
        Returns:
        --------
        List[np.ndarray]
            Predicted observations
        """
        pass
    
    def on_observation_jacobians(self, landmark_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements observation Jacobians ∂h_i/∂x and ∂h_i/∂y_i.
        
        Parameters:
        -----------
        landmark_idx : int
            Index of the landmark
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - Hx: Jacobian w.r.t. vehicle state
            - Hy: Jacobian w.r.t. landmark state
        """
        self._user_didnt_implement_jacobian = True
        Hx = np.zeros((self.observation_size, self.vehicle_size))
        Hy = np.zeros((self.observation_size, self.feature_size))
        return Hx, Hy
    
    def on_observation_jacobians_numeric_increments(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return increments for numeric Jacobian estimation.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - Vehicle state increments
            - Feature state increments
        """
        veh_inc = np.full(self.vehicle_size, 1e-6)
        feat_inc = np.full(self.feature_size, 1e-6) if self.feature_size > 0 else np.array([])
        return veh_inc, feat_inc
    
    def on_subtract_observation_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes A = A - B, accounting for topology (e.g., angle wrapping).
        
        Parameters:
        -----------
        a : np.ndarray
            First observation vector
        b : np.ndarray
            Second observation vector
            
        Returns:
        --------
        np.ndarray
            Difference vector
        """
        return a - b
    
    def on_inverse_observation_model(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Implements the inverse observation model for new landmarks.
        
        Parameters:
        -----------
        observation : np.ndarray
            Observation vector
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, bool]
            - feature_mean: New feature position
            - jacobian_vehicle: ∂y_n/∂x_v
            - jacobian_obs: ∂y_n/∂h_n
            - use_jacobian: Whether to use the observation Jacobian
        """
        raise NotImplementedError("Inverse observation model not implemented")
    
    def on_new_landmark_added_to_map(self, obs_index: int, feature_index: int) -> None:
        """
        Called when a new landmark is added to the map.
        
        Parameters:
        -----------
        obs_index : int
            Index of the observation
        feature_index : int
            Index of the new feature in the state vector
        """
        pass
    
    def on_normalize_state_vector(self) -> None:
        """
        Normalize the state vector (e.g., keep angles in [-π, π]).
        """
        pass
    
    def on_post_iteration(self) -> None:
        """
        Called after each Kalman filter iteration.
        """
        pass
    
    def run_one_kalman_iteration(self) -> None:
        """
        Execute one complete Kalman filter iteration: prediction + update.
        """
        self.time_logger.tic("KF_iteration")
        
        try:
            # Step 1: Prediction step
            self._prediction_step()
            
            # Step 2: Update step
            self._update_step()
            
            # Step 3: Post-processing
            self.on_normalize_state_vector()
            self.on_post_iteration()
            
        finally:
            self.time_logger.toc("KF_iteration")
    
    def _prediction_step(self) -> None:
        """Execute the prediction step of the Kalman filter."""
        self.time_logger.tic("KF_prediction")
        
        # Get action vector
        action = self.on_get_action()
        
        # Predict vehicle state
        vehicle_state = self.state_vector[:self.vehicle_size].copy()
        predicted_state, skip_prediction = self.on_transition_model(action, vehicle_state)
        
        if not skip_prediction:
            # Update vehicle state
            self.state_vector[:self.vehicle_size] = predicted_state
            
            # Get transition Jacobian
            if self.kf_options.use_analytic_transition_jacobian:
                F = self.on_transition_jacobian()
            else:
                F = self._estimate_transition_jacobian_numeric(action, vehicle_state)
            
            # Get process noise
            Q = self.on_transition_noise()
            
            # Update covariance matrix
            # P = F * P * F^T + Q
            F_full = np.eye(len(self.state_vector))
            F_full[:self.vehicle_size, :self.vehicle_size] = F
            
            self.covariance_matrix = (F_full @ self.covariance_matrix @ F_full.T + 
                                    self._expand_process_noise(Q))
        
        self.time_logger.toc("KF_prediction")
    
    def _update_step(self) -> None:
        """Execute the update step of the Kalman filter."""
        self.time_logger.tic("KF_update")
        
        # Get observation noise
        R = self.on_get_observation_noise()
        
        # Predict observations for existing landmarks
        if not self.is_map_empty:
            self._all_predictions = self.on_observation_model(list(range(self.number_of_landmarks)))
            self._predict_landmark_indices = self.on_pre_computing_predictions(self._all_predictions)
        else:
            self._all_predictions = []
            self._predict_landmark_indices = []
        
        # Compute innovation covariance for selected landmarks
        if self._predict_landmark_indices:
            self._compute_innovation_covariance(R)
        else:
            self._innovation_covariance = np.zeros((0, 0))
        
        # Get observations and data association
        observations, data_association = self.on_get_observations_and_data_association(
            self._all_predictions, self._innovation_covariance, 
            self._predict_landmark_indices, R)
        
        # Process observations
        if observations:
            self._process_observations(observations, data_association, R)
        
        self.time_logger.toc("KF_update")
    
    def _compute_innovation_covariance(self, R: np.ndarray) -> None:
        """Compute the innovation covariance matrix S."""
        n_landmarks = len(self._predict_landmark_indices)
        if n_landmarks == 0:
            self._innovation_covariance = np.zeros((0, 0))
            return
        
        # Initialize matrices
        H_full = np.zeros((n_landmarks * self.observation_size, len(self.state_vector)))
        
        # Compute observation Jacobians
        for i, lm_idx in enumerate(self._predict_landmark_indices):
            if self.kf_options.use_analytic_observation_jacobian:
                Hx, Hy = self.on_observation_jacobians(lm_idx)
            else:
                Hx, Hy = self._estimate_observation_jacobians_numeric(lm_idx)
            
            # Fill in the full Jacobian matrix
            obs_start = i * self.observation_size
            obs_end = obs_start + self.observation_size
            
            # Vehicle part
            H_full[obs_start:obs_end, :self.vehicle_size] = Hx
            
            # Landmark part
            if self.feature_size > 0:
                lm_start = self.vehicle_size + lm_idx * self.feature_size
                lm_end = lm_start + self.feature_size
                H_full[obs_start:obs_end, lm_start:lm_end] = Hy
        
        # Compute innovation covariance: S = H * P * H^T + R_full
        R_full = np.kron(np.eye(n_landmarks), R)
        HP_HT = H_full @ self.covariance_matrix @ H_full.T
        
        # Ensure dimensions match
        if HP_HT.shape != R_full.shape:
            self.logger.warning(f"Dimension mismatch: HP_HT {HP_HT.shape} vs R_full {R_full.shape}")
            # Resize R_full to match
            min_dim = min(HP_HT.shape[0], R_full.shape[0])
            R_full_resized = np.zeros_like(HP_HT)
            R_full_resized[:min_dim, :min_dim] = R_full[:min_dim, :min_dim]
            R_full = R_full_resized
        
        self._innovation_covariance = HP_HT + R_full
    
    def _process_observations(self, observations: List[np.ndarray], 
                            data_association: List[int], R: np.ndarray) -> None:
        """Process the observations and update the state."""
        # Add new landmarks first
        self._add_new_landmarks(observations, data_association, R)
        
        # Update with associated observations
        associated_obs = []
        associated_predictions = []
        H_matrix_rows = []
        
        for i, (obs, assoc_idx) in enumerate(zip(observations, data_association)):
            if assoc_idx >= 0 and assoc_idx < len(self._all_predictions):
                associated_obs.append(obs)
                associated_predictions.append(self._all_predictions[assoc_idx])
                
                # Get Jacobian for this observation
                if self.kf_options.use_analytic_observation_jacobian:
                    Hx, Hy = self.on_observation_jacobians(assoc_idx)
                else:
                    Hx, Hy = self._estimate_observation_jacobians_numeric(assoc_idx)
                
                # Create full Jacobian rows (one for each observation dimension)
                for obs_dim in range(self.observation_size):
                    H_row = np.zeros(len(self.state_vector))
                    H_row[:self.vehicle_size] = Hx[obs_dim, :]
                    
                    if self.feature_size > 0:
                        lm_start = self.vehicle_size + assoc_idx * self.feature_size
                        lm_end = lm_start + self.feature_size
                        H_row[lm_start:lm_end] = Hy[obs_dim, :]
                    
                    H_matrix_rows.append(H_row)
        
        if associated_obs:
            self._kalman_update(associated_obs, associated_predictions, H_matrix_rows, R)
    
    def _kalman_update(self, observations: List[np.ndarray], predictions: List[np.ndarray],
                      H_rows: List[np.ndarray], R: np.ndarray) -> None:
        """Perform the Kalman update step."""
        # Stack observations and predictions
        z = np.concatenate(observations)
        h = np.concatenate(predictions)
        H = np.vstack(H_rows)
        
        # Innovation - compute pairwise then concatenate
        innovations = []
        for obs, pred in zip(observations, predictions):
            innov = self.on_subtract_observation_vectors(obs, pred)
            innovations.append(innov)
        innovation = np.concatenate(innovations)
        
        # Innovation covariance
        n_obs = len(observations)
        R_full = np.kron(np.eye(n_obs), R)
        HP_HT = H @ self.covariance_matrix @ H.T
        
        # Ensure dimensions match
        if HP_HT.shape != R_full.shape:
            self.logger.warning(f"Kalman update dimension mismatch: HP_HT {HP_HT.shape} vs R_full {R_full.shape}")
            # Resize R_full to match
            min_dim = min(HP_HT.shape[0], R_full.shape[0])
            R_full_resized = np.zeros_like(HP_HT)
            R_full_resized[:min_dim, :min_dim] = R_full[:min_dim, :min_dim]
            R_full = R_full_resized
        
        S = HP_HT + R_full
        
        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
            K = self.covariance_matrix @ H.T @ S_inv
        except np.linalg.LinAlgError:
            self.logger.warning("Innovation covariance is singular, using pseudo-inverse")
            S_inv = np.linalg.pinv(S)
            K = self.covariance_matrix @ H.T @ S_inv
        
        # Update state and covariance
        self.state_vector += K @ innovation
        I_KH = np.eye(len(self.state_vector)) - K @ H
        # Use Joseph form update for numerical stability
        self.covariance_matrix = I_KH @ self.covariance_matrix @ I_KH.T + K @ R_full @ K.T
    
    def _add_new_landmarks(self, observations: List[np.ndarray], 
                          data_association: List[int], R: np.ndarray) -> None:
        """Add new landmarks to the map."""
        new_landmarks = [(i, obs) for i, (obs, assoc) in enumerate(zip(observations, data_association)) 
                        if assoc == -1]
        
        if not new_landmarks or self.feature_size == 0:
            return
        
        for obs_idx, observation in new_landmarks:
            # Get inverse observation model
            try:
                feat_mean, jac_vehicle, jac_obs, use_jac = self.on_inverse_observation_model(observation)
                
                # Expand state vector
                old_size = len(self.state_vector)
                new_state = np.zeros(old_size + self.feature_size)
                new_state[:old_size] = self.state_vector
                new_state[old_size:] = feat_mean
                
                # Expand covariance matrix
                new_cov = np.zeros((old_size + self.feature_size, old_size + self.feature_size))
                new_cov[:old_size, :old_size] = self.covariance_matrix
                
                # Compute new landmark covariance
                if use_jac:
                    # Vehicle contribution
                    vehicle_cov = self.covariance_matrix[:self.vehicle_size, :self.vehicle_size]
                    feat_cov_vehicle = jac_vehicle @ vehicle_cov @ jac_vehicle.T
                    
                    # Observation noise contribution
                    feat_cov_obs = jac_obs @ R @ jac_obs.T
                    
                    # Total landmark covariance
                    feat_cov = feat_cov_vehicle + feat_cov_obs
                    
                    # Cross-covariances
                    cross_cov = jac_vehicle @ self.covariance_matrix[:self.vehicle_size, :]
                    
                    new_cov[old_size:, old_size:] = feat_cov
                    new_cov[old_size:, :old_size] = cross_cov
                    new_cov[:old_size, old_size:] = cross_cov.T
                
                # Update state and covariance
                self.state_vector = new_state
                self.covariance_matrix = new_cov
                
                # Notify derived class
                feature_idx = self.number_of_landmarks - 1
                self.on_new_landmark_added_to_map(obs_idx, feature_idx)
                
            except NotImplementedError:
                self.logger.warning(f"Cannot add new landmark: inverse observation model not implemented")
    
    def _expand_process_noise(self, Q: np.ndarray) -> np.ndarray:
        """Expand process noise to full state dimension."""
        full_Q = np.zeros((len(self.state_vector), len(self.state_vector)))
        full_Q[:Q.shape[0], :Q.shape[1]] = Q
        return full_Q
    
    def _estimate_transition_jacobian_numeric(self, action: np.ndarray, 
                                            state: np.ndarray) -> np.ndarray:
        """Estimate transition Jacobian numerically."""
        increments = self.on_transition_jacobian_numeric_increments()
        jacobian = np.zeros((self.vehicle_size, self.vehicle_size))
        
        # Nominal prediction
        nominal_pred, _ = self.on_transition_model(action, state)
        
        for i in range(self.vehicle_size):
            # Perturb state
            perturbed_state = state.copy()
            perturbed_state[i] += increments[i]
            
            # Compute perturbed prediction
            perturbed_pred, _ = self.on_transition_model(action, perturbed_state)
            
            # Finite difference
            jacobian[:, i] = (perturbed_pred - nominal_pred) / increments[i]
        
        return jacobian
    
    def _estimate_observation_jacobians_numeric(self, landmark_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate observation Jacobians numerically."""
        veh_inc, feat_inc = self.on_observation_jacobians_numeric_increments()
        
        # Nominal prediction
        nominal_pred = self.on_observation_model([landmark_idx])[0]
        
        # Vehicle Jacobian
        Hx = np.zeros((self.observation_size, self.vehicle_size))
        for i in range(self.vehicle_size):
            # Perturb vehicle state
            old_vehicle = self.state_vector[:self.vehicle_size].copy()
            self.state_vector[i] += veh_inc[i]
            
            # Compute perturbed prediction
            perturbed_pred = self.on_observation_model([landmark_idx])[0]
            
            # Finite difference
            Hx[:, i] = (perturbed_pred - nominal_pred) / veh_inc[i]
            
            # Restore state
            self.state_vector[:self.vehicle_size] = old_vehicle
        
        # Feature Jacobian
        Hy = np.zeros((self.observation_size, self.feature_size))
        if self.feature_size > 0:
            lm_start = self.vehicle_size + landmark_idx * self.feature_size
            lm_end = lm_start + self.feature_size
            
            for i in range(self.feature_size):
                # Perturb landmark state
                old_landmark = self.state_vector[lm_start:lm_end].copy()
                self.state_vector[lm_start + i] += feat_inc[i]
                
                # Compute perturbed prediction
                perturbed_pred = self.on_observation_model([landmark_idx])[0]
                
                # Finite difference
                Hy[:, i] = (perturbed_pred - nominal_pred) / feat_inc[i]
                
                # Restore state
                self.state_vector[lm_start:lm_end] = old_landmark
        
        return Hx, Hy
