import numpy as np
from PIL import Image
import random
import os
import csv

def generate_heightmap(filename, csv_filename, size=513, world_size_meters=100.0, num_craters=40):
    # Map resolution
    meters_per_pixel = world_size_meters / size
    print(f"Resolution: {meters_per_pixel:.3f} m/pixel")
    
    # Initialize flat ground at middle (128)
    heightmap = np.ones((size, size), dtype=np.float32) * 128.0

    craters = []

    # Add craters
    random.seed(42) # Deterministic generation for reproducibility
    np.random.seed(42)

    for i in range(num_craters):
        # Random position in pixel coords
        cx_px = random.randint(0, size-1)
        cy_px = random.randint(0, size-1)
        
        # Radius in pixels (5m to 25m diameter -> 2.5m to 12.5m radius)
        # 2.5m / 0.2m/px = 12.5 px
        # 12.5m / 0.2m/px = 62.5 px
        radius_px = random.randint(12, 60) 
        
        # Store ground truth (convert to world coordinates)
        # Image origin (0,0) is usually top-left.
        # Gazebo (0,0) is center of image.
        # X is right, Y is up (or down depending on convention).
        # Typically: u (col) -> x, v (row) -> y.
        # Gazebo heightmap center is (0,0).
        # x_world = (cx_px - size/2) * m_per_px
        # y_world = (cy_px - size/2) * m_per_px (check sign)
        # Let's assume standard image mapping:
        # x = (col - center) * scale
        # y = (center - row) * scale (if y is up)
        
        x_world = (cx_px - size/2.0) * meters_per_pixel
        # In image, y increases downwards. In ros/gazebo world frame (map), y is typically 'North' (up on screen).
        # So low pixel Y = high World Y.
        y_world = (size/2.0 - cy_px) * meters_per_pixel
        
        r_world = radius_px * meters_per_pixel
        
        craters.append({'id': i, 'x': x_world, 'y': y_world, 'r': r_world})
        
        # Create a grid of coordinates for drawing
        y, x = np.ogrid[-radius_px: radius_px, -radius_px: radius_px]
        mask = x**2 + y**2 <= radius_px**2
        
        # Calculate crater depth profile (bowl shape)
        dist_sq = x**2 + y**2
        depth_scale = 30.0 # moderate depth in pixel values (0-255 scale)
        depth_profile = depth_scale * (1 - dist_sq / radius_px**2)
        
        # Calculate mask coordinates in the global map
        y_min, y_max = max(0, cy_px-radius_px), min(size, cy_px+radius_px)
        x_min, x_max = max(0, cx_px-radius_px), min(size, cx_px+radius_px)
        
        # Extract relative mask parts
        mask_y_start = max(0, - (cy_px-radius_px))
        mask_y_end = mask.shape[0] - max(0, (cy_px+radius_px)-size)
        mask_x_start = max(0, - (cx_px-radius_px))
        mask_x_end = mask.shape[1] - max(0, (cx_px+radius_px)-size)
        
        mask_h = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        profile_h = depth_profile[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        
        if mask_h.shape[0] > 0 and mask_h.shape[1] > 0:
            region = heightmap[y_min:y_max, x_min:x_max]
            diff = np.zeros_like(region)
            diff[mask_h] = profile_h[mask_h]
            region -= diff
            heightmap[y_min:y_max, x_min:x_max] = region

    # Clip values
    heightmap = np.clip(heightmap, 0, 255).astype(np.uint8)

    # Save Image
    img = Image.fromarray(heightmap)
    img.save(filename)
    print(f"Generated clean heightmap at {filename}")
    
    # Save CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['id', 'x', 'y', 'r']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for c in craters:
            writer.writerow(c)
    print(f"Saved crater ground truth to {csv_filename}")

if __name__ == "__main__":
    img_path = "/home/pb/Documents/R2/src/lunar_nav/worlds/lunar_heightmap.png"
    csv_path = "/home/pb/Documents/R2/src/lunar_nav/config/craters_ground_truth.csv"
    
    # Ensure config dir exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    generate_heightmap(img_path, csv_path, size=513, world_size_meters=100.0, num_craters=15)
