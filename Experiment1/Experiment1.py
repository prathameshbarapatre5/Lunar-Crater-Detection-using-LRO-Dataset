import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from lunar_nav.msg import CraterArray
import csv
import time
import math
import sys
import numpy as np
import tf_transformations

class ExperimentRunner(Node):
    def __init__(self, filename='experiment_data.csv'):
        super().__init__('experiment_runner')
        
        self.output_filename = filename
        # Data Buffers
        self.rows = []
        
        self.start_time = time.time()
        
        # Buffers for synchronization
        self.last_gt = None
        self.last_gt_yaw = 0.0
        self.last_odom = None
        self.last_pf = None
        self.last_pf_yaw = 0.0
        self.last_q = 0.0
        self.last_craters_count = 0
        
        # Subscribers
        self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/pf_pose', self.pf_callback, 10)
        self.create_subscription(Float32, '/q_score', self.q_callback, 10)
        self.create_subscription(CraterArray, '/detected_craters', self.crater_callback, 10)
        
        # Publisher for goal
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Experiment Logic
        self.create_timer(0.1, self.loop) # 10Hz logging
        self.step = 0
        
        # Waypoints for a square trajectory 20m x 20m centered roughly around craters
        # Ensure these are reachable and not inside obstacles
        # Waypoints shifted to avoid crater at (4.8, 1.3) with r=5.1
        # Start at (-10, -10) which is safe
        self.waypoints = [
            (-10.0, -10.0),
            (10.0, -10.0),
            (10.0, 10.0),
            (-10.0, 10.0),
            (-10.0, -10.0)
        ]
        self.current_wp = 0
        self.wp_threshold = 1.0
        
        self.get_logger().info(f"Experiment Runner Started. Saving to {self.output_filename}")

    def gt_callback(self, msg):
        self.last_gt = msg
        q = msg.pose.pose.orientation
        _, _, self.last_gt_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

    def odom_callback(self, msg):
        self.last_odom = msg

    def pf_callback(self, msg):
        self.last_pf = msg
        q = msg.pose.pose.orientation
        _, _, self.last_pf_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

    def q_callback(self, msg):
        self.last_q = msg.data

    def crater_callback(self, msg):
        self.last_craters_count = len(msg.craters)

    def send_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        self.goal_pub.publish(msg)

    def loop(self):
        t = time.time() - self.start_time
        
        if self.last_gt:
            # 1. State Machine for Waypoints
            gx, gy = self.waypoints[self.current_wp]
            curr_x = self.last_gt.pose.pose.position.x
            curr_y = self.last_gt.pose.pose.position.y
            dist = math.hypot(gx - curr_x, gy - curr_y)
            
            if dist < self.wp_threshold:
                self.get_logger().info(f"Reached Waypoint {self.current_wp}: ({gx}, {gy})")
                self.current_wp += 1
                if self.current_wp >= len(self.waypoints):
                    self.get_logger().info("Experiment Completed!")
                    self.save_data()
                    rclpy.shutdown()
                    return
            
            # Continuously send goal to ensure Navigator has it
            if self.step % 20 == 0: # 0.5 Hz
                self.send_goal(gx, gy)
        
        # 2. Logging
        if self.last_gt and self.last_odom: # Wait for init
            row = {
                'timestamp': t,
                'gt_x': self.last_gt.pose.pose.position.x,
                'gt_y': self.last_gt.pose.pose.position.y,
                'gt_yaw': self.last_gt_yaw,
                'odom_x': self.last_odom.pose.pose.position.x,
                'odom_y': self.last_odom.pose.pose.position.y,
                'q_score': self.last_q,
                'craters': self.last_craters_count
            }
            
            # PF
            if self.last_pf:
                row['pf_x'] = self.last_pf.pose.pose.position.x
                row['pf_y'] = self.last_pf.pose.pose.position.y
                row['pf_yaw'] = self.last_pf_yaw
                # Get PF covariance (variance in x and y)
                cov = self.last_pf.pose.covariance
                # 0:xx, 7:yy
                row['pf_cov_x'] = cov[0]
                row['pf_cov_y'] = cov[7]
            else:
                row['pf_x'] = ""
                row['pf_y'] = ""
                row['pf_yaw'] = ""
                row['pf_cov_x'] = ""
                row['pf_cov_y'] = ""
            
            self.rows.append(row)
            
        self.step += 1

    def save_data(self):
        # Allow absolute path or relative
        path = self.output_filename
        if not path.startswith('/'):
             path = '/home/pb/Documents/R2/' + path
             
        fieldnames = ['timestamp', 'gt_x', 'gt_y', 'gt_yaw', 'odom_x', 'odom_y', 'pf_x', 'pf_y', 'pf_yaw', 'pf_cov_x', 'pf_cov_y', 'q_score', 'craters']
        
        try:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            self.get_logger().info(f"Data saved to {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save data: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    # Check for output filename argument
    # args might contain known_args and remaps. simple sys.argv check is fragile if ROS args present.
    # But usually sys.argv[1] is the first non-ros arg if passed correctly.
    # Let's assume user passes it.
    output_filename = 'experiment_data.csv'
    # Iterate to find first arg that doesn't start with -- or key:=value
    # Simplified: just look at last arg if it looks like a filename
    if len(sys.argv) > 1:
        potential_name = sys.argv[-1]
        if not potential_name.startswith('--') and ':=' not in potential_name:
             output_filename = potential_name
        
    experiment = ExperimentRunner(output_filename)
    
    try:
        rclpy.spin(experiment)
    except KeyboardInterrupt:
        pass
    finally:
        # experiment.save_data() # Already called in loop on completion, but safe to call again if interrupted
        experiment.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
