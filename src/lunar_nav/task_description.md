# ShadowNav Project Proposal

## Objective
Implement the particle filter-based localization framework from ShadowNav (Atha et al., IEEE Robotics and Automation Letters, 2024), addressing autonomous global positioning for lunar rovers using crater landmarks.

## Scenario
Autonomous lunar south pole navigation where GPS is unavailable. Long-range traverses (200m+) requiring global localization to avoid hazards.

## System Components

### 1. Mathematical Formulation
*   **State**: $x_t = [p_x, p_y, \theta]^T$ (2D pose)
*   **Motion Model**: $x_t = f(x_{t-1}, u_t) + w_t$ (Odometry with ~2% drift)
*   **Measurements**: $z_t = \{(c_i, r_i)\}_{i=1}^N$ (Crater centroids and radii)
*   **Map**: $m = \{(c_j, r_j)\}_{j=1}^M$ (Orbital DEM data, 0.25-0.5m/px resolution)
*   **Algorithm**: Sequential Monte Carlo (Particle Filter) to estimate posterior $p(x_t | z_{1:t}, u_{1:t}, m)$.

### 2. Experiments
1.  **Localization Accuracy**: Compare Particle Filter ATE vs Odometry. Expected 0.5-1.5m accuracy.
2.  **Sensitivity Analysis**: Vary particle count and crater density.
3.  **Robustness**: Test against noise, wheel slip, and detection dropout.
4.  **End-to-End Navigation**: Waypoint navigation using A* and Pure Pursuit.

## Implementation Plan
1.  **Environment**: Gazebo Classic 11 with custom lunar heightmap (procedural craters).
2.  **Robot**: Differential drive rover with Depth Camera and IMU.
3.  **Nodes**:
    *   `crater_detector`: Extract crater rims from depth image.
    *   `particle_filter`: Fuse odom and visual detections.
    *   `navigator`: A* path planning on occupancy grid.

## Success Criteria
*   ATE < 2m.
*   Real-time operation.
*   Crater detection rate > 60%.
*   Successful autonomous traverses.
