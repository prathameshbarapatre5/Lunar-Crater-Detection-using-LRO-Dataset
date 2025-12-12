# Lunar Crater Navigation and Terrain-Relative Localization
## Vision-based Particle Filter Localization/Navigation

**Lunar Crater Navigation** is a particle-filter based localization and navigation system designed for GPS-denied lunar environments. It uses a depth camera to detect craters and triangulate the rover's position against a known orbital map, fusing these measurements with wheel odometry to achieve robust tracking.

![System Diagram](./system_diagram.md)

### Features
*   **Crater Detection**: Identifies craters in depth imagery using Hough Circle Transform.
*   **Particle Filter**: Fuses odometry and visual observations for global localization.
*   **Active Navigation**: A* path planning + Pure Pursuit controller capable of driving solely based on the filter's estimate.
*   **Evaluation Metrics**: Real-time ATE (Absolute Trajectory Error), Q-Score (Map Match Quality), and Heading Error logging.

### Installation
This package is built for ROS 2 Humble.

```bash
# Clone the repository
git clone <repo_url>
cd lunar_nav

# Build the package
colcon build --packages-select lunar_nav
source install/setup.bash
```

### Usage

#### 1. Launch Simulation & Localization
Launches Gazebo, the rover model, and the full perception/localization stack.
```bash
ros2 launch lunar_nav crater_navigation.launch.py
```

#### 2. Run Experiments
Execute a predefined autonomous mission to collect performance data.
```bash
python3 src/lunar_nav/scripts/run_experiment.py
```

#### 3. Analyze Results
Generate performance plots (Trajectory, ATE, Q-Score) from the experiment log.
```bash
python3 analyze_results.py experiment_data.csv
```

### System Architecture
The system consists of four main nodes:
1.  **`crater_detector`**: Processes `/camera/depth/image_raw` to find craters (x, y, r).
2.  **`particle_filter`**: Maintains a belief distribution of the robot's pose.
3.  **`navigator`**: Plans A* paths on an occupancy grid and generates velocity commands.
4.  **`experiment_runner`**: Supervisor node that sequences waypoints and logs Ground Truth vs. Estimated states.

### Results
*   **Accuracy**: 3.6m RMSE (Simulated Lunar Surface)
*   **Heading Error**: < 0.5 degrees
*   **Update Rate**: 10 Hz (Odom), ~2 Hz (Vision Correction)

### License
MIT License
