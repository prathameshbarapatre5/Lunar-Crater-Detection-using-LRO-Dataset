import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from nav_msgs.msg import Odometry
from lunar_nav.msg import CraterArray
import numpy as np
import tf_transformations
import csv
import os
import math
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parameters
        self.declare_parameter('n_particles', 200)
        self.declare_parameter('map_file', '')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        
        self.num_particles = self.get_parameter('n_particles').value
        
        # State: [x, y, theta]
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Map: List of (x, y, r)
        self.map_craters = []
        self.load_map()
        
        # Initialization state
        self.is_initialized = False
        self.last_odom_time = None
        self.last_odom_pose = None # [x, y, theta]
        
        # Motion Model Noise (std dev)
        self.odom_alpha1 = 0.05 # rotation from rotation
        self.odom_alpha2 = 0.05 # rotation from translation
        self.odom_alpha3 = 0.05 # translation from translation
        self.odom_alpha4 = 0.05 # translation from rotation
        
        # Sensor Model Noise
        self.sigma_range = 2.0  # meters (Tightened for precision)
        self.sigma_bearing = 0.1 # radians
        self.sigma_radius = 1.0 # meters
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.crater_sub = self.create_subscription(CraterArray, 'detected_craters', self.crater_callback, 10)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'pf_pose', 10)
        self.particle_pub = self.create_publisher(PoseArray, 'particle_cloud', 10)
        self.q_score_pub = self.create_publisher(Float32, 'q_score', 10)
        
        # Helper for TF
        # self.tf_buffer = ... (Optional if we need rigorous TF lookups, but we'll assume odom frame consistency for now)
        
        self.timer = self.create_timer(0.5, self.publish_particles)

        self.get_logger().info('Particle Filter Node Started')

    def load_map(self):
        # Default path relative to src if not parameters
        # In install, share directory is used.
        # For simplicity, we hardcode the source path dev environment or user user param.
        # Ideally, use package share.
        
        csv_path = "/home/pb/Documents/R2/src/lunar_nav/config/craters_ground_truth.csv"
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                # Calibration: -23 degrees (approx -0.4 rad) to fix systematic heading error
                calib_theta = -0.4
                c, s = np.cos(calib_theta), np.sin(calib_theta)
                
                for row in reader:
                    # Original swap logic
                    raw_x = float(row['y'])
                    raw_y = float(row['x'])
                    
                    # Apply Rotation
                    rot_x = raw_x * c - raw_y * s
                    rot_y = raw_x * s + raw_y * c
                    
                    self.map_craters.append({
                        'x': rot_x,
                        'y': rot_y,
                        'r': float(row['r'])
                    })
            self.get_logger().info(f"Loaded {len(self.map_craters)} craters from map.")
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {e}")

    def normalize_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def odom_callback(self, msg):
        # Extract pose
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        current_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        if not self.is_initialized:
            # Local Initialization (High Precision)
            # Now that map orientation is verified, we lock to the known start.
            ix = self.get_parameter('initial_x').value
            iy = self.get_parameter('initial_y').value
            ith = self.get_parameter('initial_yaw').value
            
            self.particles[:, 0] = np.random.normal(ix, 0.5, self.num_particles)
            self.particles[:, 1] = np.random.normal(iy, 0.5, self.num_particles)
            self.particles[:, 2] = np.random.normal(ith, 0.1, self.num_particles)
            self.last_odom_pose = current_pose
            self.is_initialized = True
            return

        # Calculate odometry delta
        dx = current_pose[0] - self.last_odom_pose[0]
        dy = current_pose[1] - self.last_odom_pose[1]
        dtheta = self.normalize_angle(current_pose[2] - self.last_odom_pose[2])
        
        # Inverse motion model (Odometry sampled motion model)
        # Calculate relative motion steps: rot1, trans, rot2
        trans = math.sqrt(dx**2 + dy**2)
        # Avoid division by zero
        if trans < 0.001:
             rot1 = 0
             rot2 = dtheta
        else:
             rot1 = math.atan2(dy, dx) - self.last_odom_pose[2]
             rot2 = dtheta - rot1
             
        rot1 = self.normalize_angle(rot1)
        rot2 = self.normalize_angle(rot2)
        
        # Update particles
        # Sample noisy motion
        # rot1_hat = rot1 - sample(alpha1*rot1 + alpha2*trans)
        # trans_hat = trans - sample(alpha3*trans + alpha4*(rot1+rot2))
        # rot2_hat = rot2 - sample(alpha1*rot2 + alpha2*trans)
        
        scale_rot1 = self.odom_alpha1 * abs(rot1) + self.odom_alpha2 * trans
        scale_trans = self.odom_alpha3 * trans + self.odom_alpha4 * (abs(rot1) + abs(rot2))
        scale_rot2 = self.odom_alpha1 * abs(rot2) + self.odom_alpha2 * trans
        
        noise_rot1 = np.random.normal(0, scale_rot1 + 1e-6, self.num_particles)
        noise_trans = np.random.normal(0, scale_trans + 1e-6, self.num_particles)
        noise_rot2 = np.random.normal(0, scale_rot2 + 1e-6, self.num_particles)
        
        r1 = rot1 - noise_rot1
        tr = trans - noise_trans
        r2 = rot2 - noise_rot2
        
        self.particles[:, 0] += tr * np.cos(self.particles[:, 2] + r1)
        self.particles[:, 1] += tr * np.sin(self.particles[:, 2] + r1)
        self.particles[:, 2] += r1 + r2
        self.particles[:, 2] = np.vectorize(self.normalize_angle)(self.particles[:, 2])

        self.last_odom_pose = current_pose
        # Always publish the Prediction Update (Dead Reckoning)
        # This prevents the Navigator from stalling if no craters are currently visible.
        self.publish_estimated_pose(msg.header)

    def crater_callback(self, msg):
        if not self.is_initialized or len(msg.craters) == 0:
            return
            
        # Measurement Update
        # For each particle, calculate likelihood of observed craters matching map craters
        
        for i in range(self.num_particles):
            px, py, ptheta = self.particles[i]
            w = 1.0
            
            # transform all measurements to global frame based on this particle hypothesis
            # Observation z is relative to robot (camera frame usually forward/right/down)
            # wait, crater_detector output converts to 'Camera Frame' (x right, y down, z forward)
            # Rover Base Frame: X forward, Y left, Z up.
            # We need to handle frame transform.
            # Assuming Camera is mounted facing Forward.
            # x_cam (right) -> -y_rover
            # y_cam (down) -> -z_rover
            # z_cam (forward) -> x_rover
            # Let's approximate: 
            # obs_x_rover = obs_z_cam
            # obs_y_rover = -obs_x_cam
            
            for z in msg.craters:
                # Convert camera frame to rover frame
                # Camera: Z is depth (forward). X is right.
                rover_dist = z.z
                rover_bearing_local = math.atan2(-z.x, z.z) # Revert to Standard Frame (-x is left)
                
                # Global pos of the crater according to particle
                # obs_global_x = px + rover_dist * cos(ptheta + rover_bearing_local)
                # obs_global_y = py + rover_dist * sin(ptheta + rover_bearing_local)
                
                obs_global_x = px + rover_dist * np.cos(ptheta + rover_bearing_local)
                obs_global_y = py + rover_dist * np.sin(ptheta + rover_bearing_local)
                obs_r = z.radius
                
                # Data Association: Find nearest neighbor using BOTH Position and Radius
                # Mahalanobis-like distance: D = sqrt( (dx/sigma_p)^2 + (dy/sigma_p)^2 + (dr/sigma_r)^2 )
                best_score = float('inf')
                best_crater = None
                
                for mc in self.map_craters:
                    dx = mc['x'] - obs_global_x
                    dy = mc['y'] - obs_global_y
                    dr = mc['r'] - obs_r
                    
                    # Weighted distance metric
                    # Sigma ranges (tuning): Pos=0.5m, Radius=0.5m
                    score = (dx/self.sigma_range)**2 + (dy/self.sigma_range)**2 + (dr/self.sigma_radius)**2
                    
                    if score < best_score:
                        best_score = score
                        best_crater = mc
                
                # Chi-Square 3-DOF approx threshold (95% conf) ~ 7.8
                # If score > threshold, it's an outlier (clutter)
                if best_crater and best_score < 15.0:  # Relaxed threshold for robustness
                     # Calculate Probability
                     # error is already proportional to score
                     prob = np.exp(-0.5 * best_score)
                     w *= prob + 1e-9
                else:
                    w *= 1e-9 # Outlier penalty

            self.weights[i] = w

        # Normalize weights
        sum_w = np.sum(self.weights)
        if sum_w > 0:
            self.weights /= sum_w
        else:
            self.weights.fill(1.0/self.num_particles)
            
        # Adaptive Resampling (N_eff)
        # 1 / sum(w^2)
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-9)
        
        # Only resample if effective particles < 50%
        if n_eff < self.num_particles / 2.0:
            new_particles = np.zeros_like(self.particles)
            r = np.random.uniform(0, 1.0/self.num_particles)
            c = self.weights[0]
            i = 0
            for m in range(self.num_particles):
                U = r + m * (1.0/self.num_particles)
                while U > c:
                    i = (i + 1) % self.num_particles
                    c += self.weights[i]
                new_particles[m] = self.particles[i]
            
            self.particles = new_particles
            self.weights.fill(1.0 / self.num_particles)
        # Else: Keep current particles and weights (Sequential Importance Sampling)

        # Calculate and Publish Q-score for the current estimate
        mean_x = np.mean(self.particles[:, 0])
        mean_y = np.mean(self.particles[:, 1])
        mean_theta = np.arctan2(np.sum(np.sin(self.particles[:, 2])), np.sum(np.cos(self.particles[:, 2])))
        
        q_score = self.calculate_q_score(mean_x, mean_y, mean_theta, msg.craters)
        
        q_msg = Float32()
        q_msg.data = float(q_score)
        self.q_score_pub.publish(q_msg)

        self.publish_estimated_pose(msg.header)

    def publish_particles(self):
        msg = PoseArray()
        msg.header.frame_id = 'odom' # We display relative to Odom frame for visualization overlay, or 'map' if we had a map-odom transform
        # For this setup: The particles are estimating Pose in the MAP frame (implied by the csv coordinates)
        # But we don't have a map->odom transform broadcaster here.
        # So we just visualize them in the 'odom' frame assuming odom frame ~= map frame at start (0,0)
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Decimate for performance if needed
        for i in range(0, self.num_particles, 2):
            p = Pose()
            p.position.x = self.particles[i, 0]
            p.position.y = self.particles[i, 1]
            q = tf_transformations.quaternion_from_euler(0, 0, self.particles[i, 2])
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            msg.poses.append(p)
        self.particle_pub.publish(msg)

    def publish_estimated_pose(self, header):
        mean_x = np.mean(self.particles[:, 0])
        mean_y = np.mean(self.particles[:, 1])
        # Circular mean for Theta
        mean_theta = np.arctan2(np.sum(np.sin(self.particles[:, 2])), np.sum(np.cos(self.particles[:, 2])))
        
        # Calculate Covariance
        cov = np.zeros((3, 3))
        # Errors
        dx = self.particles[:, 0] - mean_x
        dy = self.particles[:, 1] - mean_y
        dtheta = self.particles[:, 2] - mean_theta
        # Normalize angle differences
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # Stack
        err = np.vstack((dx, dy, dtheta))
        cov = np.cov(err)
        
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = mean_x
        msg.pose.pose.position.y = mean_y
        q = tf_transformations.quaternion_from_euler(0, 0, mean_theta)
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Populate 6x6 covariance (row-major)
        # indices: 0:x, 1:y, 5:theta (yaw)
        # msg.pose.covariance is float64[36]
        c = np.zeros(36)
        c[0]  = cov[0, 0] # xx
        c[1]  = cov[0, 1] # xy
        c[5]  = cov[0, 2] # x-yaw
        c[6]  = cov[1, 0] # yx
        c[7]  = cov[1, 1] # yy
        c[11] = cov[1, 2] # y-yaw
        c[30] = cov[2, 0] # yaw-x
        c[31] = cov[2, 1] # yaw-y
        c[35] = cov[2, 2] # yaw-yaw
        
        msg.pose.covariance = c.tolist()
        
        self.pose_pub.publish(msg)
        
        # Calculate Q-Score (Match Quality) for the Mean Estimate
        # Q = 1 / (Sum of Errors + epsilon) normalized
        # We need the current observations 'z' to compute this against the mean estimate
        # Since this function is called inside crater_callback (implied context), we need access to 'msg.craters'
        # But 'msg' is not passed here. 
        # Refactor: Return mean pose from this function or calculate Q in crater_callback.
        
    def calculate_q_score(self, mean_x, mean_y, mean_theta, obstacles):
        # Q-score logic based on ShadowNav (Inverse of error)
        total_error = 0.0
        matches = 0
        
        for z in obstacles:
             # Transform obs to global using mean estimate
             rover_dist = z.z
             rover_bearing = math.atan2(-z.x, z.z)
             
             gx = mean_x + rover_dist * np.cos(mean_theta + rover_bearing)
             gy = mean_y + rover_dist * np.sin(mean_theta + rover_bearing)
             r = z.radius
             
             # Nearest neighbor in map
             best_dist = float('inf')
             best_map = None
             for mc in self.map_craters:
                 d = math.sqrt((mc['x'] - gx)**2 + (mc['y'] - gy)**2)
                 if d < best_dist:
                     best_dist = d
                     best_map = mc

             if best_map and best_dist < 5.0: # Threshold
                 # Error metric: Euclidean distance + Radius difference
                 # The paper says "distance between observed crater edge and ground truth"
                 # Simple proxy: center distance
                 total_error += best_dist
                 matches += 1
             else:
                 # Penalty for unmatched feature
                 total_error += 5.0 
        
        if matches == 0 and len(obstacles) > 0:
             return 0.0
        
        # Q = 1 / (mean_error + epsilon)
        if matches > 0:
            mean_error = total_error / len(obstacles) # Average error per feature
            epsilon = 0.1
            q = 1.0 / (mean_error + epsilon)
            # Normalize? If error is 0, q -> 10. If error is 1m, q -> 0.9. 
            # Let's clip to [0, 1] for typical scaling or leave as raw score.
            # Paper suggests 0.4 is good. 
            # If Q_inc was sum of errors, Q = 1/Q_inc.
            # If Q_inc was 2.5 (avg error 2.5m?), Q = 0.4.
            return min(q, 1.0)
        else:
            return 0.0

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
