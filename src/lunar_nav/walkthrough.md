# Lunar Crater Navigation: Project Walkthrough

This document outlines the setup, architecture, and execution steps for the **Lunar Crater Navigation** (Terrain-Relative Localization) project. It simulates a rover navigating a crater-rich lunar south pole environment, using an orbital map and onboard depth sensors to localize itself without GPS.

## Architecture

1.  **Simulation Environment** (Gazebo):
    *   **Terrain**: High-resolution lunar heightmap with procedural craters (bowl-shaped).
    *   **Rover**: 4-wheel differential drive rover equipped with Depth Camera and IMU.
    *   **Map**: `craters_ground_truth.csv` generated alongside the terrain.

2.  **Perception stack** (`crater_detector.py`):
    *   **Input**: Depth images (`/camera/depth/image_raw`).
    *   **Processing**:
        *   Hough Circle Transform for rim extraction (optimized for depth maps).
        *   Deprojection from 2D pixels to 3D Rover-Frame coordinates ($x,y,z$) using camera intrinsics.
    *   **Output**: `/detected_craters` (List of observed landmarks in metric space).

3.  **State Estimation** (`particle_filter.py`):
    *   **Algorithm**: Sequential Monte Carlo (SMC) Particle Filter.
    *   **Map**: Loads known crater locations from CSV.
    *   **Metric**: Calculates **Q-Score** (Match Quality) as defined in Lunar Crater Navigation literature.
    *   **Output**: `/pf_pose` (Global estimated pose with covariance), `/particle_cloud` (Visual uncertainty), `/q_score`.

4.  **Autonomous Navigation** (`navigator.py`):
    *   **Algorithm**: A* Path Planning + Pure Pursuit Controller.
    *   **Map**: Occupancy grid built from Ground Truth CSV.
    *   **Input**: `/goal_pose` (Target destination).

5.  **Teleoperation** (`teleop_keyboard.py`):
    *   **Functionality**: Manual control of the rover for testing and data collection.
    *   **Input**: Keyboard keystrokes.
    *   **Output**: `/cmd_vel` (Velocity commands).
    *   **Physics Safety**: Implements gradual acceleration to prevent flipping.


```bash
cd ~/ros2_ws
colcon build --packages-select lunar_nav
source install/setup.bash
```

## Running Experiments

### 1. Launch Simulation
This starts Gazebo, spawns the rover safely, and starts all nodes (Detector, PF, Navigator).
```bash
ros2 launch lunar_nav crater_navigation.launch.py
```

### 2. Autonomous Control
To make the rover drive autonomously to a specific point (e.g., x=10, y=10):

**Using Terminal:**
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 10.0, y: 10.0, z: 0.0}}}" --once
```

**Using RViz:**
1.  Open RViz: `ros2 run rviz2 rviz2`
2.  Set **Fixed Frame** to `map`.
3.  Add **Map** (Topic: `/map`, if available) or just visualize **RobotModel**.
4.  Add **Path** (Topic: `/planned_path`) to see the A* plan.
5.  Use the **2D Nav Goal** tool in the top toolbar. Click and drag on the map to set a target. The rover should start moving.

### 3. Automated Data Collection and Analysis
To faithfully reproduce the Lunar Crater Navigation results, follow the procedures below for the distinct experiments.

#### Experiment 1: Localization Accuracy (Standard Run)
**Goal:** Measure ATE and Heading Error under nominal conditions (N=200 particles).

1.  **Terminal 1 (Simulation)**:
    ```bash
    ros2 launch lunar_nav crater_navigation.launch.py
    ```
2.  **Terminal 2 (Experiment Runner)**:
    ```bash
    source install/setup.bash
    # This sends waypoints and logs data to ~/Documents/R2/experiment_data.csv
    python3 src/lunar_nav/scripts/run_experiment.py
    ```
    *Wait for the rover to complete the square trajectory (~2 minutes).*
3.  **Terminal 3 (Analysis)**:
    After the experiment loop finishes or is stopped (Ctrl+C):
    ### 3.3 Analyze Results
 
 We provide a script to generate plots (Trajectory, ATE, Q-Score) and a summary report.
 
 ```bash
 python3 analyze_results.py experiment_data.csv
 ```
 
 This will print the performance metrics (ATE, Mean Heading Error) and save `experiment_results.png`.
    *   **Output**: `experiment_results.png` (Plots: Trajectory, ATE, Heading Error, Uncertainty).
    *   **Report**: Performance metrics printed to console.

#### Experiment 2: Sensitivity Analysis (Varied Particles)
**Goal:** Determine how particle count affects accuracy.

**Run 1 (N=100):**
1.  Edit `src/lunar_nav/launch/crater_navigation.launch.py`: Set `'n_particles': 100`.
2.  Rebuild: `colcon build --packages-select lunar_nav`.
3.  **Terminal 1**: Launch Simulation.
    ```bash
    ros2 launch lunar_nav crater_navigation.launch.py
    ```
4.  **Terminal 2**: Run Experiment Interactively.
    ```bash
    source install/setup.bash
    python3 Experiment2/experiment2.py
    # PROMPT: Enter output filename: n100
    ```
5.  **Terminal 3**: Analyze.
    ```bash
    python3 Experiment2/analysis2.py Experiment2/n100.csv
    ```

**Run 2 (N=500):**
1.  Edit Launch File: Set `'n_particles': 500`.
2.  Rebuild & Launch.
3.  Run `Experiment2/experiment2.py` -> Enter `n500`.
4.  Analyze `n500.csv`.

**Run 3 (N=2000):**
1.  Edit Launch File: Set `'n_particles': 2000`.
2.  Rebuild & Launch.
3.  Run `Experiment2/experiment2.py` -> Enter `n2000`.
4.  Analyze `n2000.csv`.

#### Experiment 4: End-to-End Navigation (Active Localization)
**Goal:** Verify the rover can navigate using *only* the Particle Filter estimate (Active Control).

1.  **Terminal 1 (Simulation with PF Control)**:
    ```bash
    # Ensure n_particles is 2000 (Optimal config)
    ros2 launch lunar_nav crater_navigation.launch.py use_pf:=True
    ```
2.  **Terminal 2 (Experiment Runner)**:
    ```bash
    source install/setup.bash
    python3 src/lunar_nav/scripts/run_experiment.py
    ```
3.  **Analysis**:
    Run `analyze_results.py` on the output CSV.
    *   **Success Criteria**: The rover completes the square path without crashing or getting stuck.
    *   **Plot Check**: The "Particle Filter" trajectory should look smooth (as the controller follows it), while the "Ground Truth" trajectory will show the *actual* path. If PF is accurate, they will overlap. If PF diverges, the Ground Truth will drift away from the square.

## Monitoring & Visualization

**Check Localization:**
Watch the estimated pose and confidence (covariance).
```bash
ros2 topic echo /pf_pose
ros2 topic echo /q_score
```

**Visualize Particles in RViz:**
*   **Fixed Frame**: `map` or `odom`.
*   **Add** -> **PoseArray** -> Topic: `/particle_cloud`.
*   You will see a cluster of arrows representing the filter's belief of the rover's position. As the rover sees craters, the arrows should converge.

**Visualize Craters:**
*   **Add** -> **pointCloud2** (if available) or check `crater_debug_image` in `rqt_image_view`.

## Troubleshooting
*   **Rover flips over?** Ensure `spawn_z` is high enough (currently 1.0m).
*   **No Path Found?** The goal might be inside a crater (obstacle). Try a different point.
*   **"Frame [map] does not exist"?** We have added a static transform publisher to the launch file. Ensure you sourced the workspace. If using an older build, use `odom` as Fixed Frame.
