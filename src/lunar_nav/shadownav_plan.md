# ShadowNav Implementation & Execution Plan

## 1. System Implementation Status

### A. Environment
*   **Terrain**: Procedurally generated Lunar Heightmap (8-bit grayscale PNG) with 15-20 bowl-shaped craters.
*   **Ground Truth**: `craters_ground_truth.csv` contains exact $(x, y, r)$ of all craters.
*   **Gazebo World**: `lunar_crater.world` uses the heightmap and P3D plugin for ground truth tracking.

### B. Robot
*   **Model**: Differential drive rover with 4 wheels (skid-steer style).
*   **Sensors**:
    *   Depth Camera (`/camera/depth/image_raw`).
    *   IMU (`/imu`).
    *   Odometry (`/odom`).
*   **Modifications**: Low Center of Mass ($z=-0.1$) and soft suspension to prevent flipping.

### C. Software Stack
*   **`crater_detector.py`**:
    *   Hough Circle Transform on depth images.
    *   3D Deprojection using Camera Matrix ($K$).
    *   Publishes `/detected_craters`.
*   **`particle_filter.py`**:
    *   200 Particles, SMC algorithm.
    *   Loads `craters_ground_truth.csv`.
    *   Updates weights based on observation likelihood.
    *   Computes **Q-Score** (0-1 match quality).
    *   Publishes `/pf_pose` and `/particle_cloud`.
*   **`navigator.py`**:
    *   A* Path Planning on 100x100m grid.
    *   Pure Pursuit Controller.
    *   Obstacle avoidance using safety margins.

## 2. Execution Instructions

### A. Build the Package
```bash
cd ~/ros2_ws
colcon build --packages-select lunar_nav
source install/setup.bash
```

### B. Launch Simulation
This brings up Gazebo, the Rover, and all Navigation Nodes.
```bash
ros2 launch lunar_nav crater_navigation.launch.py
```
*Wait ~10 seconds for Gazebo to load.*

### C. Operation Modes

#### Mode 1: Manual Teleoperation
Open a new terminal:
```bash
source install/setup.bash
ros2 run lunar_nav teleop_keyboard
```
*   Keys: `W/S` (Drive), `A/D` (Turn).
*   Observe `/pf_pose` converging in RViz as you drive near craters.

#### Mode 2: Autonomous Navigation
Send a target 2D pose:
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 10.0, y: 10.0, z: 0.0}}}" --once
```
The rover will plan a path (Green line in RViz) and drive to it.

#### Mode 3: Experiment & Data Collection
Run the automated experiment runner which drives a 20m square pattern and logs data.
1.  Ensure Simulation is running.
2.  Run Script:
    ```bash
    source install/setup.bash
    python3 src/lunar_nav/scripts/run_experiment.py
    ```
3.  **Output**: `experiment_data.csv` in `~/Documents/R2/`.

### D. Analysis
The CSV contains:
*   `gt_x, gt_y`: Ground Truth Position.
*   `pf_x, pf_y`: Particle Filter Estimate.
*   `odom_x, odom_y`: Raw Odometry.
*   `q_score`: Detection quality.
*   `pf_cov_x, pf_cov_y`: Uncertainty.

**Metrics to Calculate**:
1.  **ATE (RMSE)**: $\sqrt{\frac{1}{N} \sum (pf_i - gt_i)^2}$.
2.  **Detection Rate**: Average craters detected per frame.
