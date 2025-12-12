#!/usr/bin/env python3
"""
ROS 2 Keyboard Teleop Node
Location: ~/Documents/R2/src/lunar_nav/lunar_nav/teleop_keyboard.py

Controls the rover using keyboard for testing.
Keys:
    w/s: forward/backward
    a/d: turn left/right
    q/e: forward-left/forward-right
    space: stop
    +/-: increase/decrease speed
    Ctrl-C: quit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty


class TeleopKeyboard(Node):
    """Keyboard teleoperation node"""
    
    def __init__(self):
        super().__init__('teleop_keyboard')
        
        # Publisher
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Speed settings
        self.linear_speed = 0.4 # Reduced for safety
        self.angular_speed = 1.0
        self.current_linear = 0.0
        self.current_angular = 0.0  # rad/s
        
        # Key mappings
        self.move_bindings = {
            'w': (1, 0),   # Forward
            's': (-1, 0),  # Backward
            'a': (0, 1),   # Turn left
            'd': (0, -1),  # Turn right
            'q': (1, 1),   # Forward left
            'e': (1, -1),  # Forward right
        }
        
        self.speed_bindings = {
            '+': 1.1,  # Increase speed
            '-': 0.9,  # Decrease speed
        }
        
        # Terminal settings
        self.settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info('Teleop Keyboard Node initialized')
        self.print_instructions()
    
    def print_instructions(self):
        """Print control instructions"""
        msg = """
=== Lunar Rover Teleop ===
Controls:
  w/s: forward/backward
  a/d: turn left/right
  q/e: forward+left/forward+right
  +/-: increase/decrease speed
  space: stop
  Ctrl-C: quit

Current settings:
  Linear speed:  {:.2f} m/s
  Angular speed: {:.2f} rad/s
""".format(self.linear_speed, self.angular_speed)
        print(msg)
    
    def get_key(self):
        """Get keyboard input"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def print_status(self):
        """Print current speed settings"""
        print(f"\rLinear: {self.linear_speed:.2f} m/s | Angular: {self.angular_speed:.2f} rad/s     ", end='', flush=True)
    
    def run(self):
        """Main loop"""
        try:
            while rclpy.ok():
                key = self.get_key()
                
                # Check if key is in move bindings
                if key in self.move_bindings.keys():
                    x = self.move_bindings[key][0]
                    th = self.move_bindings[key][1]
                    # Smooth values
                    self.target_linear = x * self.linear_speed
                    self.target_angular = th * self.angular_speed
                    
                    # Simple Low pass filter / Ramp
                    self.current_linear = 0.5 * self.current_linear + 0.5 * self.target_linear
                    self.current_angular = 0.5 * self.current_angular + 0.5 * self.target_angular
                    
                    twist = Twist()
                    twist.linear.x = float(self.current_linear)
                    twist.angular.z = float(self.current_angular)
                    
                    self.pub.publish(twist)
                # Check if key is in speed bindings
                elif key in self.speed_bindings.keys():
                    self.linear_speed = self.linear_speed * self.speed_bindings[key]
                    self.angular_speed = self.angular_speed * self.speed_bindings[key]
                    self.print_status()
                    # No velocity command is published for speed changes, continue to next loop iteration
                    continue
                
                # Space to stop
                elif key == ' ':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                
                # Ctrl-C to quit
                elif key == '\x03':
                    break
                
                else:
                    # Unknown key, send stop command
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                
                # Publish twist
                self.pub.publish(twist)
        
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
        
        finally:
            # Stop the robot
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            print("\nTeleop stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboard()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
