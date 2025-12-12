import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from lunar_nav.msg import Crater, CraterArray

class CraterDetector(Node):
    def __init__(self):
        super().__init__('crater_detector')
        
        # Parameters
        self.declare_parameter('min_radius', 10)
        self.declare_parameter('max_radius', 100)
        self.declare_parameter('canny_threshold', 50)
        self.declare_parameter('accumulator_threshold', 25)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.crater_pub = self.create_publisher(CraterArray, 'detected_craters', 10)
        self.debug_image_pub = self.create_publisher(Image, 'crater_debug_image', 10)
        
        self.bridge = CvBridge()
        self.camera_model = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        self.get_logger().info('Crater Detector Node Started')

    def camera_info_callback(self, msg):
        if self.fx is None:
            # K is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            k = np.array(msg.k).reshape(3, 3)
            self.fx = k[0, 0]
            self.fy = k[1, 1]
            self.cx = k[0, 2]
            self.cy = k[1, 2]
            self.get_logger().info(f'Camera Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def image_callback(self, msg):
        if self.fx is None:
            return  # Wait for calibration

        try:
            # Convert ROS Image to OpenCV
            # Handle float32 depth (meters) or 16UC1 (mm) or bgr8 (visual)
            # Assuming standard simulation output is often 32FC1
            if msg.encoding == '32FC1':
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                # Replace NaNs
                depth_img = np.nan_to_num(depth_img)
                
                # Create visual 8-bit image for Hough
                # Normalize 0-10m to 0-255
                norm_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                cv_mono = np.uint8(norm_img)
                
            elif 'bgr' in msg.encoding or 'rgb' in msg.encoding:
                # Fallback if gazebo works in color mode
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_mono = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                # In this case depth is fake, linear mapping 0-255
                # We can't easily recover meters unless we assume min/max range
                depth_img = np.float32(cv_mono) / 255.0 * 50.0 # Approximation using far clip 50m
                
            else:
                 self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
                 return

            # Improve contrast
            cv_mono = cv2.equalizeHist(cv_mono)
            # Blur
            cv_mono = cv2.medianBlur(cv_mono, 5)
            
            # Params
            min_r = self.get_parameter('min_radius').value
            max_r = self.get_parameter('max_radius').value
            canny_t = self.get_parameter('canny_threshold').value
            accum_t = self.get_parameter('accumulator_threshold').value
            
            circles = cv2.HoughCircles(
                cv_mono, 
                cv2.HOUGH_GRADIENT, 
                dp=1.2, 
                minDist=30,
                param1=canny_t,
                param2=accum_t,
                minRadius=min_r,
                maxRadius=max_r
            )
            
            crater_msg = CraterArray()
            crater_msg.header = msg.header # Keep camera frame
            
            debug_img = cv2.cvtColor(cv_mono, cv2.COLOR_GRAY2BGR)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    u, v, r_px = i[0], i[1], i[2]
                    
                    # Ensure within bounds
                    if v < depth_img.shape[0] and u < depth_img.shape[1]:
                        # Get depth at centroid
                        z_meters = float(depth_img[v, u])
                        
                        # Fix for RGB fallback case where 0 is close (black) or far?
                        # Usually depth map: dark=close, bright=far? Or opposite?
                        # In Gazebo visual depth: usually dark=close.
                        
                        # Project to 3D Camera Frame
                        # X (right), Y (down), Z (forward)
                        x_meters = (u - self.cx) * z_meters / self.fx
                        y_meters = (v - self.cy) * z_meters / self.fy
                        
                        # Estimate physical radius
                        # Similar triangles: r_meters / z_meters = r_px / fx
                        r_meters = (r_px * z_meters) / self.fx
                        
                        # Draw
                        cv2.circle(debug_img, (u, v), r_px, (0, 255, 0), 2)
                        
                        c = Crater()
                        c.x = x_meters
                        c.y = y_meters
                        c.z = z_meters # Distance forward
                        c.radius = r_meters
                        c.confidence = 1.0
                        crater_msg.craters.append(c)
            
            self.crater_pub.publish(crater_msg)
            
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CraterDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
