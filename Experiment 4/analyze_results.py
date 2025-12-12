import csv
import math
import sys
import matplotlib.pyplot as plt

def calculate_ate(gt_data, est_data):
    """
    Calculate Absolute Trajectory Error (RMSE)
    """
    if len(gt_data) != len(est_data):
        print("Warning: Data length mismatch for ATE calculation")
        return 0.0
    
    squared_error_sum = 0.0
    for i in range(len(gt_data)):
        dx = gt_data[i][0] - est_data[i][0]
        dy = gt_data[i][1] - est_data[i][1]
        squared_error_sum += (dx*dx + dy*dy)
    
    rmse = math.sqrt(squared_error_sum / len(gt_data))
    return rmse

def normalize_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

def calculate_heading_error(gt_yaw, est_yaw):
    errors = []
    for i in range(len(gt_yaw)):
        diff = abs(normalize_angle(gt_yaw[i] - est_yaw[i]))
        errors.append(diff)
    return errors

def analyze(csv_file):
    timestamps = []
    gt = []
    gt_yaws = []
    odom = []
    pf = []
    pf_yaws = []
    pf_covs = []
    q_scores = []
    craters = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row['timestamp'])
                    gt_x = float(row['gt_x'])
                    gt_y = float(row['gt_y'])
                    gt_yaw = float(row.get('gt_yaw', 0.0))
                    odom_x = float(row['odom_x'])
                    odom_y = float(row['odom_y'])
                    q = float(row['q_score'])
                    # Handle PF potentially empty/nan
                    pf_x_str = row['pf_x']
                    pf_y_str = row['pf_y']
                    pf_yaw_str = row.get('pf_yaw', "")
                    c = int(row['craters'])
                    
                    if pf_x_str and pf_x_str.lower() != 'nan' and pf_x_str != "":
                        pf_x = float(pf_x_str)
                        pf_y = float(pf_y_str)
                        pf.append((pf_x, pf_y))
                        if pf_yaw_str:
                             pf_yaws.append(float(pf_yaw_str))
                        else:
                             pf_yaws.append(0.0)
                        
                        # Covariance
                        cov_x = float(row.get('pf_cov_x', 0.0) or 0.0)
                        cov_y = float(row.get('pf_cov_y', 0.0) or 0.0)
                        pf_covs.append(math.sqrt(cov_x + cov_y)) # Approx std dev magnitude
                    else:
                        pf.append((None, None))
                        pf_yaws.append(None)
                        pf_covs.append(None)
                        
                    timestamps.append(t)
                    gt.append((gt_x, gt_y))
                    gt_yaws.append(gt_yaw)
                    odom.append((odom_x, odom_y))
                    q_scores.append(q)
                    craters.append(c)
                except (ValueError, TypeError) as e:
                    # print(f"Row error: {e}, Row: {row}")
                    continue
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

    # Filter out None PFs for ATE
    valid_gt = []
    valid_gt_yaw = []
    valid_pf = []
    valid_pf_yaw = []
    valid_pf_cov = []
    valid_t = []
    valid_q = []
    
    for i in range(len(pf)):
        if pf[i][0] is not None:
            valid_gt.append(gt[i])
            valid_gt_yaw.append(gt_yaws[i])
            valid_pf.append(pf[i])
            valid_pf_yaw.append(pf_yaws[i])
            valid_pf_cov.append(pf_covs[i])
            valid_t.append(timestamps[i])
            valid_q.append(q_scores[i])
    
    # Calculate Metrics
    if len(valid_gt) == 0:
        print("No valid PF data points found.")
        return

    ate_pf = calculate_ate(valid_gt, valid_pf)
    ate_odom = calculate_ate(valid_gt, odom[:len(valid_gt)]) 
    
    heading_errors = calculate_heading_error(valid_gt_yaw, valid_pf_yaw)
    mean_heading_error = sum(heading_errors) / len(heading_errors) if heading_errors else 0.0
    
    print("========================================")
    print("ShadowNav Performance Report")
    print("========================================")
    print(f"Total Duration: {timestamps[-1] - timestamps[0]:.2f} s")
    print(f"Data Points: {len(timestamps)}")
    print(f"Particle Filter ATE: {ate_pf:.3f} m")
    print(f"Odometry ATE: {ate_odom:.3f} m")
    print(f"Mean Heading Error: {math.degrees(mean_heading_error):.2f} deg")
    if ate_odom > 0:
        print(f"Improvement: {(ate_odom - ate_pf)/ate_odom * 100:.1f}%")
    else:
        print("Improvement: N/A")
    print(f"Avg Q-Score: {sum(q_scores)/len(q_scores):.3f}")
    print(f"Avg Craters Detected: {sum(craters)/len(craters):.1f}")
    print("========================================")
    
    # PLOTTING
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    
    # 1. Trajectory
    ax = axs[0, 0]
    gt_x = [p[0] for p in gt]
    gt_y = [p[1] for p in gt]
    odom_x = [p[0] for p in odom]
    odom_y = [p[1] for p in odom]
    pf_x = [p[0] for p in valid_pf]
    pf_y = [p[1] for p in valid_pf]
    
    ax.plot(gt_x, gt_y, 'k-', label='Ground Truth', linewidth=2)
    ax.plot(odom_x, odom_y, 'r--', label='Odometry')
    ax.plot(pf_x, pf_y, 'b.-', label='Particle Filter', markersize=2, alpha=0.5)
    ax.set_title('Trajectory Comparison')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # 2. Error over time
    ax = axs[0, 1]
    err_pf = [math.sqrt((valid_gt[i][0]-valid_pf[i][0])**2 + (valid_gt[i][1]-valid_pf[i][1])**2) for i in range(len(valid_pf))]
    # Assume odom aligns for plot
    err_odom = [math.sqrt((gt[i][0]-odom[i][0])**2 + (gt[i][1]-odom[i][1])**2) for i in range(len(valid_pf))]
    
    ax.plot(valid_t, err_pf, 'b-', label='PF Error')
    ax.plot(valid_t, err_odom, 'r--', label='Odom Error')
    ax.set_title('Localization Error (ATE) over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.legend()
    ax.grid(True)
    
    # 3. Q-Score
    ax = axs[1, 0]
    ax.plot(valid_t, valid_q, 'g-')
    ax.plot([valid_t[0], valid_t[-1]], [0.4, 0.4], 'k--', label='Threshold (0.4)')
    ax.set_title('Q-Score (Match Quality)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Q-Score (0-1)')
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    ax.legend()
    
    # 4. Craters
    ax = axs[1, 1]
    ax.plot(timestamps, craters, 'm-')
    ax.set_title('Detected Craters per Frame')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Count')
    ax.grid(True)
    
    # 5. Heading Error
    ax = axs[2, 0]
    ax.plot(valid_t, [math.degrees(e) for e in heading_errors], 'r-')
    ax.set_title('Heading Error (deg)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (deg)')
    ax.grid(True)
    
    # 6. Particle Spread (Covariance)
    ax = axs[2, 1]
    ax.plot(valid_t, valid_pf_cov, 'b-', label='Positional Uncertainty')
    ax.set_title('Particle Filter Uncertainty (Covariance)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Std Dev (m)')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/pb/Documents/R2/experiment_results.png')
    print("Plots saved to /home/pb/Documents/R2/experiment_results.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        analyze('/home/pb/Documents/R2/experiment_data.csv')
