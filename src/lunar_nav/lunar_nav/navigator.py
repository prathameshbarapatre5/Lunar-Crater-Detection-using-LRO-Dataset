import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
import csv
import math
import heapq
import numpy as np
import tf_transformations

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        # Parameters
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('resolution', 0.5) # meters per cell
        self.declare_parameter('world_size', 100.0) # meters
        self.declare_parameter('safety_margin', 1.0) # meters buffer around craters
        self.declare_parameter('use_pf', False) # Use PF for feedback?
        
        self.resolution = self.get_parameter('resolution').value
        self.world_size = self.get_parameter('world_size').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.use_pf = self.get_parameter('use_pf').value
        
        self.grid_size = int(self.world_size / self.resolution)
        self.offset = self.world_size / 2.0
        
        # Obstacle Map
        self.obstacles = []
        self.load_obstacles()
        
        # Occupancy Grid (0: Free, 1: Obstacle)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.generate_grid_map()
        
        # State
        self.current_pose = None # [x, y, yaw]
        self.path = [] # List of (x, y) tuples
        self.current_wp_index = 0
        self.goal_reached = False
        
        # Pub/Sub
        if self.use_pf:
            self.pf_sub = self.create_subscription(PoseWithCovarianceStamped, '/pf_pose', self.pf_callback, 10)
            self.get_logger().warn("Navigator using PARTICLE FILTER for control.")
        else:
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
            self.get_logger().info("Navigator using ODOMETRY for control.")

        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        # Control Loop
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Navigator Node Started. Waiting for goal...')

    def load_obstacles(self):
        csv_path = "/home/pb/Documents/R2/src/lunar_nav/config/craters_ground_truth.csv"
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.obstacles.append({
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'r': float(row['r'])
                    })
            self.get_logger().info(f"Loaded {len(self.obstacles)} obstacles.")
        except Exception as e:
            self.get_logger().warn(f"Could not load obstacles: {e}")

    def world_to_grid(self, x, y):
        gx = int((x + self.offset) / self.resolution)
        gy = int((y + self.offset) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx * self.resolution) - self.offset
        wy = (gy * self.resolution) - self.offset
        return wx, wy

    def generate_grid_map(self):
        # Mark obstacles
        for obs in self.obstacles:
            # Bounding box in grid
            r_eff = obs['r'] + self.safety_margin
            x_min, y_min = self.world_to_grid(obs['x'] - r_eff, obs['y'] - r_eff)
            x_max, y_max = self.world_to_grid(obs['x'] + r_eff, obs['y'] + r_eff)
            
            # Clamp
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.grid_size-1, x_max)
            y_max = min(self.grid_size-1, y_max)
            
            # Fill circle
            r_sq = (r_eff / self.resolution)**2
            cx, cy = self.world_to_grid(obs['x'], obs['y'])
            
            for i in range(x_min, x_max+1):
                for j in range(y_min, y_max+1):
                    if (i - cx)**2 + (j - cy)**2 <= r_sq:
                        self.grid[j, i] = 1 # Occupied (Row=Y, Col=X)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def pf_callback(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def goal_callback(self, msg):
        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        self.get_logger().info(f"Received goal: ({target_x}, {target_y})")
        
        if self.current_pose:
            self.plan_path((self.current_pose[0], self.current_pose[1]), (target_x, target_y))

    def plan_path(self, start, goal):
        sx, sy = self.world_to_grid(start[0], start[1])
        gx, gy = self.world_to_grid(goal[0], goal[1])
        
        if not (0 <= sx < self.grid_size and 0 <= sy < self.grid_size and 0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
            self.get_logger().error("Start or Goal out of bounds")
            return

        if self.grid[gy, gx] == 1:
            self.get_logger().warn("Goal is inside an obstacle!")
            return

        # A* Algorithm
        open_set = []
        heapq.heappush(open_set, (0, sx, sy))
        
        came_from = {}
        g_score = { (sx, sy): 0 }
        f_score = { (sx, sy): math.hypot(sx-gx, sy-gy) }
        
        found = False
        
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            
            if (cx, cy) == (gx, gy):
                found = True
                break
            
            # Neighbors (8-connected)
            neighbors = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[ny, nx] == 1:
                        continue # Obstacle
                        
                    movement_cost = math.hypot(dx, dy)
                    tentative_g = g_score[(cx, cy)] + movement_cost
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = (cx, cy)
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + math.hypot(nx-gx, ny-gy)
                        f_score[(nx, ny)] = f
                        heapq.heappush(open_set, (f, nx, ny))
        
        if found:
            # Reconstruct path
            curr = (gx, gy)
            path_grid = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path_grid.append(curr)
            
            path_grid.reverse()
            # Convert to world coords
            self.path = [self.grid_to_world(p[0], p[1]) for p in path_grid]
            self.current_wp_index = 0
            self.goal_reached = False
            self.publish_path()
            self.get_logger().info(f"Path found with {len(self.path)} waypoints")
        else:
            self.get_logger().error("No path found!")

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for p in self.path:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def control_loop(self):
        if not self.path or self.goal_reached or self.current_pose is None:
            return

        # Pure Pursuit Logic
        target_wp = self.path[self.current_wp_index]
        
        dx = target_wp[0] - self.current_pose[0]
        dy = target_wp[1] - self.current_pose[1]
        dist = math.hypot(dx, dy)
        
        # Lookahead distance mechanism
        lookahead = 1.0
        while dist < lookahead and self.current_wp_index < len(self.path) - 1:
            self.current_wp_index += 1
            target_wp = self.path[self.current_wp_index]
            dx = target_wp[0] - self.current_pose[0]
            dy = target_wp[1] - self.current_pose[1]
            dist = math.hypot(dx, dy)
            
        if self.current_wp_index == len(self.path) - 1 and dist < 0.5:
            self.goal_reached = True
            self.stop_robot()
            self.get_logger().info("GOAL REACHED!")
            return

        # Calculate control
        desired_yaw = math.atan2(dy, dx)
        current_yaw = self.current_pose[2]
        
        yaw_err = desired_yaw - current_yaw
        # Normalize
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        
        twist = Twist()
        
        # Simple P-Controller
        if abs(yaw_err) > 1.0:
            # Turn in place if error is large
            twist.linear.x = 0.0
            twist.angular.z = 1.0 * np.sign(yaw_err)
        else:
            twist.linear.x = 0.8 # Constant forward speed
            twist.angular.z = 2.0 * yaw_err # Proportional turn
            
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
