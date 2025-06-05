# EKF SLAM 2D Implementation

A complete implementation of Extended Kalman Filter (EKF) based Simultaneous Localization and Mapping (SLAM) for 2D environments using range-bearing sensors.

## Features

### Core SLAM Implementation
- **EKF-based SLAM**: Complete 2D SLAM implementation using Extended Kalman Filter
- **Range-bearing sensors**: Support for LiDAR and similar sensors
- **Real-time processing**: Efficient algorithms suitable for real-time applications
- **Robust data association**: Advanced landmark matching with Mahalanobis distance
- **Uncertainty management**: Full covariance tracking for robot pose and landmarks

### Key Components
- **RangeBearingKFSLAM2D**: Main SLAM filter implementation
- **Data Association**: KDTree-based efficient landmark association
- **Perception Processing**: Point cloud clustering for landmark extraction
- **ROS Integration**: Complete ROS node for real-world deployment

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Point Cloud   │    │    Odometry      │    │   EKF SLAM      │
│   (LiDAR)       │───▶│   Processing     │───▶│   Filter        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Perception     │    │ Motion Model     │    │ Robot Pose +    │
│  Processing     │    │ (Odometry)       │    │ Map Landmarks   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│ Data            │                            │ Visualization   │
│ Association     │                            │ & TF Broadcast  │
└─────────────────┘                            └─────────────────┘
```

## Installation

### Dependencies
- Python 3.8+
- NumPy
- Matplotlib (for visualization)
- SciPy (for data association)
- ROS Noetic (for ROS node)

### Setup
```bash
# Clone the repository
git clone <repository_url>
cd SLAM

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install numpy matplotlib scipy

# For ROS integration
sudo apt install ros-noetic-tf2-ros ros-noetic-geometry-msgs ros-noetic-sensor-msgs
```

## Usage

### Standalone Testing
Run the comprehensive test suite:
```bash
cd SLAM
source .venv/bin/activate
python src/ekf_slam/test.py
```

The test suite includes:
- **Unit tests**: Individual component validation
- **Integration tests**: Full system testing
- **Convergence tests**: SLAM performance validation
- **Noise robustness tests**: Performance under different noise conditions
- **Visualization**: Automatic plot generation showing results

### ROS Integration
For real-world deployment with ROS:

```bash
# Make the script executable
chmod +x scripts/ekf_slam_node.py

# Launch the SLAM system
roslaunch slam_project ekf_slam.launch

# Or run the node directly
rosrun slam_project ekf_slam_node.py
```

### Configuration
Key parameters can be adjusted in the launch file or via ROS parameters:

#### EKF SLAM Parameters
- `process_noise_std`: Process noise [x, y, yaw] standard deviations
- `sensor_range_std`: Range measurement noise (meters)
- `sensor_bearing_std`: Bearing measurement noise (radians)

#### Data Association Parameters
- `max_association_distance`: Maximum distance for landmark association
- `mahalanobis_threshold`: Chi-squared threshold for association validation

#### Perception Parameters
- `min_cluster_size`: Minimum points per landmark cluster
- `max_cluster_size`: Maximum points per landmark cluster
- `cluster_distance_threshold`: Distance threshold for clustering
- `max_sensor_range`: Maximum sensor range

## API Reference

### Core Classes

#### `RangeBearingKFSLAM2D`
Main SLAM filter class.

```python
from ekf_slam.ekf_slam import RangeBearingKFSLAM2D, ActionCollection, SensoryFrame

# Initialize SLAM filter
slam = RangeBearingKFSLAM2D()

# Process motion and observations
action = ActionCollection(dx=1.0, dy=0.0, dyaw=0.1, timestamp=0.0)
sensor_frame = SensoryFrame(observations=[], timestamp=0.0)
slam.process_action_observation(action, sensor_frame)

# Get results
robot_pose = slam.get_current_robot_pose()
_, landmarks, landmark_ids, _, _ = slam.get_current_state()
```

#### Key Methods
- `process_action_observation()`: Main processing method
- `get_current_robot_pose()`: Get robot pose estimate
- `get_current_state()`: Get full system state
- `reset()`: Reset filter state

### Data Structures

#### `RangeBearingObservation`
```python
@dataclass
class RangeBearingObservation:
    range: float        # Distance to landmark (meters)
    yaw: float         # Bearing angle (radians)
    pitch: float       # Elevation angle (radians)
    landmark_id: int   # Landmark identifier (-1 for unknown)
```

#### `ActionCollection`
```python
@dataclass
class ActionCollection:
    dx: float          # Translation in x (meters)
    dy: float          # Translation in y (meters) 
    dyaw: float        # Rotation change (radians)
    timestamp: float   # Timestamp
    covariance: np.ndarray  # 3x3 covariance matrix
```

## Performance

### Test Results
Recent test results show excellent performance:

- **Position accuracy**: ~0.27m final error after 80 time steps
- **Landmark mapping**: 100% success rate (8/8 landmarks mapped)
- **Computational efficiency**: Real-time capable processing
- **Noise robustness**: Stable performance across noise levels

### Computational Complexity
- **Time complexity**: O(n²) where n is the number of landmarks
- **Space complexity**: O(n²) for covariance matrix storage
- **Optimizations**: Efficient Jacobian computation, selective prediction

## ROS Integration

### Topics

#### Subscribed Topics
- `/odom` (nav_msgs/Odometry): Robot odometry
- `/velodyne_points` (sensor_msgs/PointCloud2): LiDAR point cloud

#### Published Topics
- `/ekf_slam_node/robot_pose` (geometry_msgs/PoseWithCovarianceStamped): Robot pose estimate
- `/ekf_slam_node/landmarks` (visualization_msgs/MarkerArray): Landmark positions

#### TF Frames
- `map`: Global reference frame
- `odom`: Odometry frame
- `base_link`: Robot base frame
- `lidar_link`: LiDAR sensor frame

### Visualization
The system includes comprehensive RViz visualization:
- Robot pose with uncertainty ellipse
- Landmark positions as colored markers
- Point cloud data
- Coordinate frame relationships

## Technical Details

### Algorithm Overview
1. **Prediction Step**: Use odometry to predict robot motion
2. **Observation Processing**: Extract landmarks from sensor data
3. **Data Association**: Match observations to existing landmarks
4. **Update Step**: Update state estimates using Kalman filter
5. **Map Management**: Add new landmarks as needed

### Mathematical Foundation
- **State Vector**: [robot_x, robot_y, robot_yaw, lm1_x, lm1_y, lm2_x, lm2_y, ...]
- **Motion Model**: Standard odometry-based motion model
- **Observation Model**: Range-bearing sensor model with full Jacobians
- **Uncertainty**: Full covariance matrix tracking all correlations

### Key Features
- **Angle Normalization**: Proper handling of angular quantities
- **Numerical Stability**: Joseph-form covariance updates
- **Robust Association**: Mahalanobis distance-based matching
- **Efficient Computation**: Optimized matrix operations

## Troubleshooting

### Common Issues
1. **No landmarks detected**: Check sensor data quality and clustering parameters
2. **Poor localization**: Verify odometry quality and process noise settings
3. **Association failures**: Adjust association distance and threshold parameters
4. **Performance issues**: Monitor computational load and consider reducing sensor rate

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## License

[Specify your license here]

## References

- Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous localization and mapping: part I. IEEE robotics & automation magazine, 13(2), 99-110.
- Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic robotics. MIT press.
- Davison, A. J. (2003). Real-time simultaneous localisation and mapping with a single camera. ICCV 2003. 
