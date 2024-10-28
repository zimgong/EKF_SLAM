# EKF SLAM Specification Sheet

## 1. Extended Kalman Filter Algorithm

### 1.1 Problem Statement

- State vector: $\mu_t = [x, y, yaw, x_1, y_1, x_2, y_2, ... x_n, y_n]$

- Control vector: $u_t = [Ax, Ay, Ayaw]$

- Observation vector: $z_t = [range_1, yaw_1, range_2, yaw_2, ... range_m, yaw_m]$

- Non-linear transition function: $g(\mu_{t-1}, u_t)$

- Transition Jacobian: $G_t = \frac{\partial g}{\partial \mu}$

- Transition noise: $R_t$

- Non-linear observation function: $h(\mu_t)$

- Observation Jacobian: $H_t = \frac{\partial h}{\partial \mu}$

- Observation noise: $Q_t$

### 1.2 Algorithm

Extended_Kalman_Filter($\mu_{t-1}, \sum_{t-1}, u_t, z_t$): 

- Prediction Step: 
    - Prediction mean: $\bar{\mu}_t = g(\mu_{t-1}, u_t)$
    - Prediction covariance: $\bar{\sum}_t = G_t \sum_{t-1} G_t^T + R_t$

- Update:
    - $K_t = \bar{\sum}_t H_t^T (H_t \bar{\sum}_t H_t^T + Q_t)^{-1}$
    - $\mu_t = \bar{\mu}_t + K_t(z_t - h(\bar{\mu}_t))$
    - $\sum_t = (I - K_t H_t) \bar{\sum}_t$

- return $\mu_t, \sum_t$

## 2. Kalman Filters Virtual Class

See `CKalmanFilterCapable.h` for source code. 

See [Kalman Filters - MRPT](https://www.mrpt.org/Kalman_Filters) for more information. 

1. `OnGetAction`: Create action matrix $u$ from odometry. 

2. Predict new pose $xv_{k+1|k}$. 
    - `OnTransitionModel`: Implement the transition model $ \hat{x}_{k|k-1} = f(* \hat{x}_{k-1|k-2}, u_k ) $, with output $ \hat{x}_{k|k-1} $. 

3. Predict covariance $P_{k+1|k}$. 
    - `OnTransitionJacobian`: Compute the Jacobian $\frac{\partial fv}{\partial xv}$ (derivative of f_vehicle wrt x_vehicle). 
    - `OnTransitionNoise`: Define the tansition noise $Q$. 
    - Compute $Pxx$ sub-matrix. 
    - Compute all $Pxy_i$ sub-matrices. 

4. Overwrite the new state vector. 

5. Predict observations and compute Jacobians. 
    - `OnObservationModel`: Predict the observations for all the map LMs. 
    - `OnPreComputingPredictions`: Reduce the number of covariance landmark predictions to be made.

6. Compute the innovation matrix $m_S$. 
    - `OnObservationJacobian`: Implements the observation Jacobians. 
    - `OnGetObservationNoise`: Define the observation noise $R$.
    - Compute $m_S$:  $m_S = H P H' + R$. 

7. Update using the Kalman gain. 

8. Introduce new landmarks. 

## 3. EKF SLAM Implementation

See `CRangeBearingKFSLAM2D.h` for source code. 

### 3.1 Problem Statement

- VEH_SIZE: The dimension of the "vehicle state": $[x, y, yaw]$. 

- OBS_SIZE: The dimension of each observation: $[range, yaw]$. 

- FEAT_SIZE: The dimension of the features in the system state (the "map"): $[x, y]$. 

- ACT_SIZE: The dimension of each "action" u_k: $[Ax, Ay, Ayaw]$. 

### 3.2 Transition model

- Transition model: 

$$g(\mu_{t-1}, u_t) = \begin{bmatrix} x_{t-1} \\  y_{t-1} \\ yaw_{t-1} \end{bmatrix} + \begin{bmatrix} Ax \\ Ay \\ Ayaw \end{bmatrix}$$


- Transition Jacobian:

$$G_t = \begin{bmatrix} 1 & 0 & -Ax \sin(yaw_{t-1}) - Ay \cos(yaw_{t-1}) \\ 0 & 1 & Ax \cos(yaw_{t-1}) - Ay \sin(yaw_{t-1}) \\ 0 & 0 & 1 \end{bmatrix}$$

- Transition noise:

$$R_t = \begin{bmatrix} \sigma_{Ax}^2 & 0 & 0 \\ 0 & \sigma_{Ay}^2 & 0 \\ 0 & 0 & \sigma_{Ayaw}^2 \end{bmatrix}$$

### 3.3 Observation model

**TODO**: Write up the specs for the observation model.


## 4. EKF SLAM Package

See `ekf_slam.h` and `ekf_slam_wrapper.h` for source code. 

### 4.1 I/O Specifications

- Subscribers: 
    - `odom_sub`: Odometry data. $[x, y, yaw]$. 
    - `cone_sub`: Cone data. $[range, yaw]$ (Can also do $[x, y]$) with an associated ID. 

- Publisher: 
    - `tf_pub`: Publish the transformation between the map and the odom frame. 
    - `cone_pub`: Publish the map of the cones. 

## 5. Data Association

The current plan for data association is to use the nearest neighbor algorithm. 
