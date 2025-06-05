# EKF SLAM for MR24 Driverless

A complete SLAM (Simultaneous Localization and Mapping) system designed for MR24 Driverless, implemented in Python for ROS Noetic.

## Features

- **EKF SLAM**: Extended Kalman Filter-based SLAM with robust state estimation
- **KDTree Data Association**: Efficient landmark association using KDTree for fast nearest neighbor search
- **Point Cloud Processing**: Real-time processing of LiDAR point clouds for landmark detection
- **TF Integration**: Full integration with ROS TF system for coordinate frame management
- **Visualization**: Complete RViz configuration for real-time visualization
- **Racing Optimized**: Tuned parameters for high-speed autonomous racing scenarios

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Point Cloud   │───▶│   Perception     │───▶│  Data           │
│   (LiDAR)       │    │   Processor      │    │  Association    │
└─────────────────┘    └──────────────────┘    │  (KDTree)       │
                                               └─────────┬───────┘
┌─────────────────┐    ┌──────────────────┐            │
│   Odometry      │───▶│   EKF SLAM       │◀───────────┘
│   (Wheel/IMU)   │    │   Filter         │
└─────────────────┘    └─────────┬────────┘
                                 │
                       ┌─────────▼────────┐
                       │   State Output   │
                       │   (Pose + Map)   │
                       └──────────────────┘
```

## Requirements

### System Requirements
- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3.8+

### Dependencies
- numpy >= 1.19.0
- scipy >= 1.5.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- transforms3d >= 0.3.1

### ROS Dependencies
- rospy
- tf2_ros
- sensor_msgs
- geometry_msgs
- nav_msgs
- visualization_msgs

## Installation

1. **Clone the repository into your catkin workspace:**
   ```bash
   cd ~/catkin_ws/src
   git clone <repository_url> ekf_slam
   ```

2. **Install Python dependencies:**
   ```bash
   cd ekf_slam
   pip install -r requirements.txt
   ```

3. **Build the package:**
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage

### Basic Launch

Launch the complete SLAM system:
```bash
roslaunch mapping ekf_slam_2d_py.launch
```

### Custom Configuration

Launch with custom parameters:
```bash
roslaunch mapping ekf_slam_2d_py.launch \
    pointcloud_topic:=/your_lidar_topic \
    odom_topic:=/your_odom_topic \
    enable_visualization:=true
```

### Parameter Configuration

Edit the configuration file to tune parameters:
```bash
rosparam load config/slam_params.yaml
```

## Topics

### Subscribed Topics
- `/velodyne_points` (sensor_msgs/PointCloud2): LiDAR point cloud data
- `/odom` (nav_msgs/Odometry): Robot odometry for motion model

### Published Topics
- `/ekf_slam_node/robot_pose` (geometry_msgs/PoseWithCovarianceStamped): Robot pose estimate
- `/ekf_slam_node/landmarks` (visualization_msgs/MarkerArray): Landmark positions
- `/slam/status` (std_msgs/String): System status and statistics

### TF Frames
- `map` → `odom` → `base_link` → `velodyne`

## Configuration

### Key Parameters

#### EKF SLAM Parameters
```yaml
ekf_slam:
  process_noise_std: 0.1      # Motion model uncertainty
  measurement_noise_std: 0.5  # Sensor measurement uncertainty
```

#### Data Association Parameters
```yaml
data_association:
  max_association_distance: 2.0    # Maximum distance for landmark association
  mahalanobis_threshold: 9.21      # Chi-squared threshold (95% confidence)
```

#### Perception Parameters
```yaml
perception:
  min_cluster_size: 5              # Minimum points per landmark
  max_cluster_size: 50             # Maximum points per landmark
  cluster_distance_threshold: 0.5  # Clustering distance threshold
```

## Racing-Specific Features

### Cone Detection
- Optimized for Formula Student cone detection
- Configurable cone height and detection range
- Track width estimation for validation

### High-Speed Operation
- Adaptive parameters for high-speed scenarios
- Aggressive association mode for fast movement
- Optimized update rates for real-time performance

## Visualization

The system includes a complete RViz configuration showing:
- Robot pose with uncertainty ellipse
- Detected landmarks as red cylinders
- Raw point cloud data
- TF frame relationships
- Real-time system status

Launch with visualization:
```bash
roslaunch fs_slam fs_slam.launch enable_visualization:=true
```

## Troubleshooting

### Common Issues

1. **No landmarks detected:**
   - Check point cloud topic and data
   - Verify clustering parameters
   - Ensure proper TF transforms

2. **Poor localization:**
   - Tune process and measurement noise
   - Check odometry quality
   - Verify landmark associations

3. **High computational load:**
   - Reduce point cloud density
   - Increase clustering thresholds
   - Lower update rates

### Debug Mode

Enable debug logging:
```bash
rosservice call /ekf_slam_node/set_logger_level ros.fs_slam DEBUG
```

## Performance Tuning

### For High-Speed Racing
```yaml
racing:
  high_speed_mode: true
  aggressive_association: true
```

### For Accuracy
```yaml
ekf_slam:
  process_noise_std: 0.05
  measurement_noise_std: 0.3
```

## API Reference

### EKFSlam Class
```python
from fs_slam.ekf_slam import EKFSlam

slam = EKFSlam(
    process_noise_std=0.1,
    measurement_noise_std=0.5
)

# Prediction step
slam.predict(control_input=[v, omega], dt=0.1)

# Update step
slam.update(measurements=[(landmark_id, [range, bearing])])

# Get results
pose = slam.get_robot_pose()
landmarks = slam.get_landmarks()
```

### DataAssociation Class
```python
from fs_slam.data_association import DataAssociation

da = DataAssociation(
    max_association_distance=2.0,
    mahalanobis_threshold=9.21
)

# Update landmark database
da.update_landmarks(landmarks)

# Associate measurements
associations, new_measurements = da.associate_measurements(
    measurements, robot_pose, robot_covariance, measurement_covariance
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Formula Student community for racing insights
- ROS community for excellent documentation
- Open-source SLAM research community

## Contact

For questions and support, please open an issue on the repository or contact the development team. 
