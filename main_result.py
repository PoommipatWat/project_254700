import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# ==================== NUMERICAL METHODS ====================

def euler_method(f, x0, Y0, h, n_steps, x_max = None):
    x_values = [x0]
    Y_values = [Y0]
    
    for n in range(n_steps):
        x_n = x_values[n]
        Y_n = Y_values[n]
        
        Y_values.append(Y_n + h * f(x_n, Y_n))
        x_values.append(x_n + h)
        if x_max is not None and Y_values[-1][0] >= x_max:
            break
    
    return np.array(x_values), np.array(Y_values)


def improved_euler_method(f, x0, Y0, h, n_steps, x_max = None):
    x_values = [x0]
    Y_values = [Y0]
    
    for n in range(n_steps):
        x_n = x_values[n]
        Y_n = Y_values[n]
        
        # Predictor
        Y_star = Y_n + h * f(x_n, Y_n)
        
        # Corrector
        Y_values.append(Y_n + (h / 2) * (f(x_n, Y_n) + f(x_n + h, Y_star)))
        x_values.append(x_n + h)

        if x_max is not None and Y_values[-1][0] >= x_max:
            break
    
    return np.array(x_values), np.array(Y_values)


def rk4_method(f, x0, Y0, h, n_steps, x_max = None):
    x_values = [x0]
    Y_values = [Y0]
    
    for n in range(n_steps):
        x_n = x_values[n]
        Y_n = Y_values[n]
        
        k1 = h * f(x_n, Y_n)
        k2 = h * f(x_n + h/2, Y_n + k1/2)
        k3 = h * f(x_n + h/2, Y_n + k2/2)
        k4 = h * f(x_n + h, Y_n + k3)
        
        Y_values.append(Y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
        x_values.append(x_n + h)

        if x_max is not None and Y_values[-1][0] >= x_max:
            break
    
    return np.array(x_values), np.array(Y_values)


# ==================== STATE SPACE FUNCTION ====================

def F_state_space(t, Y, a):
    x = Y[0]
    v = Y[1]
    
    dx_dt = v
    dv_dt = a
    
    return np.array([dx_dt, dv_dt])


# ==================== EXACT FUNCTION ====================
def exact_solution(a, t = None, x_max = None):
    if x_max is None:
        return 0.5 * a * t**2 
    else:
        return np.sqrt(2 * x_max * (1/a))
    

# ==================== MOMENT FUNCTION ====================
def find_CG(CAR_CM_X, CAR_CM_Y, CAR_M, M_rod, rod_position):
    total_mass = CAR_M + np.sum(M_rod)

    sum_mx = CAR_CM_X * CAR_M + np.sum(M_rod * rod_position[:, 0])
    sum_my = CAR_CM_Y * CAR_M + np.sum(M_rod * rod_position[:, 1])

    CG_x = sum_mx / total_mass
    CG_y = sum_my / total_mass

    return [CG_x, CG_y]

def find_N(CG_x, CG_y, M_car, WHEEL_X, rod_position, M_rod, force, force_position):
    # Rod Moment
    r_rod_x = WHEEL_X - rod_position[:, 0]
    moment_rod = np.sum(M_rod * g * r_rod_x)

    # Car Moment
    r_car_x = WHEEL_X - CG_x
    moment_car = M_car * g * r_car_x

    # Force Moment
    r_force_x = WHEEL_X - force_position[0]
    r_force_y = force_position[1]
    moment_force = (force[1] * r_force_x) - (force[0] * r_force_y)

    r_N_x = 105.850/1000
    N_back = (moment_rod + moment_car - moment_force) / r_N_x
    N_front = (M_car + np.sum(M_rod))*g - force[1] - N_back

    return [N_front, N_back], N_front < 0 or N_back < 0


# ==================== PLOTTING FUNCTIONS ====================

def plot_trajectory_comparison(T_euler, Y_euler, T_improved, Y_improved, T_rk4, Y_rk4, exact_func, a, save_path=None):
    """
    เปรียบเทียบ trajectory ของแต่ละวิธี
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Position vs Time
    axes[0].plot(T_euler, Y_euler[:, 0], 'r--', label='Euler', linewidth=2, alpha=0.7)
    axes[0].plot(T_improved, Y_improved[:, 0], 'g--', label='Improved Euler', linewidth=2, alpha=0.7)
    axes[0].plot(T_rk4, Y_rk4[:, 0], 'b-', label='RK4', linewidth=2)
    
    # Exact solution
    T_exact = np.linspace(0, T_rk4[-1], 1000)
    X_exact = exact_func(a, t=T_exact)
    axes[0].plot(T_exact, X_exact, 'k:', label='Exact', linewidth=2)
    
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Position (m)', fontsize=12)
    axes[0].set_title('Position vs Time - Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Velocity vs Time
    axes[1].plot(T_euler, Y_euler[:, 1], 'r--', label='Euler', linewidth=2, alpha=0.7)
    axes[1].plot(T_improved, Y_improved[:, 1], 'g--', label='Improved Euler', linewidth=2, alpha=0.7)
    axes[1].plot(T_rk4, Y_rk4[:, 1], 'b-', label='RK4', linewidth=2)
    
    # Exact velocity
    V_exact = a * T_exact
    axes[1].plot(T_exact, V_exact, 'k:', label='Exact', linewidth=2)
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1].set_title('Velocity vs Time - Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_analysis(step_sizes, errors_euler, errors_improved, errors_rk4, save_path=None):
    """
    พล็อต Error vs Step Size (Convergence Test)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(step_sizes, errors_euler, 'ro-', label='Euler', linewidth=2, markersize=8)
    axes[0].plot(step_sizes, errors_improved, 'gs-', label='Improved Euler', linewidth=2, markersize=8)
    axes[0].plot(step_sizes, errors_rk4, 'b^-', label='RK4', linewidth=2, markersize=8)
    axes[0].set_xlabel('Step Size (s)', fontsize=12)
    axes[0].set_ylabel('Absolute Error (m)', fontsize=12)
    axes[0].set_title('Error vs Step Size (Linear Scale)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Log-log scale
    axes[1].loglog(step_sizes, errors_euler, 'ro-', label='Euler (1st order)', linewidth=2, markersize=8)
    axes[1].loglog(step_sizes, errors_improved, 'gs-', label='Improved Euler (2nd order)', linewidth=2, markersize=8)
    axes[1].loglog(step_sizes, errors_rk4, 'b^-', label='RK4 (4th order)', linewidth=2, markersize=8)
    
    # Reference lines
    if len(step_sizes) > 1:
        axes[1].loglog(step_sizes, errors_euler[0] * (step_sizes/step_sizes[0])**1, 'r:', alpha=0.5, label='1st order ref')
        axes[1].loglog(step_sizes, errors_improved[0] * (step_sizes/step_sizes[0])**2, 'g:', alpha=0.5, label='2nd order ref')
        axes[1].loglog(step_sizes, errors_rk4[0] * (step_sizes/step_sizes[0])**4, 'b:', alpha=0.5, label='4th order ref')
    
    axes[1].set_xlabel('Step Size (s)', fontsize=12)
    axes[1].set_ylabel('Absolute Error (m)', fontsize=12)
    axes[1].set_title('Error vs Step Size (Log-Log Scale)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_efficiency_comparison(methods, times, errors, save_path=None):
    """
    เปรียบเทียบ Computational Efficiency
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Computation Time
    axes[0].bar(methods, times, color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Computation Time (s)', fontsize=12)
    axes[0].set_title('Computation Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Error
    axes[1].bar(methods, errors, color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Absolute Error (m)', fontsize=12)
    axes[1].set_title('Accuracy (Error)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    # Efficiency (Accuracy / Time)
    efficiency = 1 / (np.array(errors) * np.array(times))
    axes[2].bar(methods, efficiency, color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Efficiency (1/(Error × Time))', fontsize=12)
    axes[2].set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_configuration_heatmap(hole_positions, rod_configs, times, overturned_mask, save_path=None):
    """
    Heatmap แสดงเวลาสำหรับแต่ละ configuration
    """
    # สร้างข้อมูลสำหรับ heatmap
    data = times.copy()
    data[overturned_mask] = np.nan  # ตำแหน่งที่คว่ำ
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # สร้าง heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Time (s)'}, 
                xticklabels=hole_positions,
                yticklabels=[f'Config {i+1}' for i in range(len(rod_configs))],
                ax=ax, linewidths=0.5, linecolor='gray',
                vmin=np.nanmin(data), vmax=np.nanmax(data))
    
    # เพิ่มสัญลักษณ์สำหรับตำแหน่งที่คว่ำ
    for i in range(len(rod_configs)):
        for j in range(len(hole_positions)):
            if overturned_mask[i, j]:
                ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center',
                       color='red', fontsize=20, fontweight='bold')
    
    ax.set_xlabel('Force Hole Position (1=lowest, 6=highest)', fontsize=12)
    ax.set_ylabel('Rod Configuration', fontsize=12)
    ax.set_title('Time to Finish Line (X = Overturned)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_force_angle_analysis(hole_positions, angles, forces, accelerations, save_path=None):
    """
    วิเคราะห์มุมแรงและผลต่อความเร่ง
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Force Angle
    axes[0, 0].plot(hole_positions, np.degrees(angles), 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Hole Position', fontsize=12)
    axes[0, 0].set_ylabel('Force Angle (degrees)', fontsize=12)
    axes[0, 0].set_title('Force Angle vs Hole Position', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Force Components
    axes[0, 1].plot(hole_positions, forces[:, 0], 'ro-', label='Horizontal (Fx)', linewidth=2, markersize=8)
    axes[0, 1].plot(hole_positions, forces[:, 1], 'go-', label='Vertical (Fy)', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Hole Position', fontsize=12)
    axes[0, 1].set_ylabel('Force (N)', fontsize=12)
    axes[0, 1].set_title('Force Components vs Hole Position', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Acceleration
    axes[1, 0].plot(hole_positions, accelerations, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Hole Position', fontsize=12)
    axes[1, 0].set_ylabel('Acceleration (m/s²)', fontsize=12)
    axes[1, 0].set_title('Acceleration vs Hole Position', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficiency (Fx/F_total)
    efficiency = forces[:, 0] / np.linalg.norm(forces, axis=1)
    axes[1, 1].plot(hole_positions, efficiency, 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Hole Position', fontsize=12)
    axes[1, 1].set_ylabel('Force Efficiency (Fx/F_total)', fontsize=12)
    axes[1, 1].set_title('Force Efficiency vs Hole Position', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_stability_analysis(configs, cg_positions, n_fronts, n_backs, save_path=None):
    """
    วิเคราะห์ความมั่นคงของรถ (ป้องกันการคว่ำ)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # CG Position
    axes[0, 0].plot(configs, cg_positions[:, 0]*1000, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=52.925, color='r', linestyle='--', label='Right Wheel Position', linewidth=2)
    axes[0, 0].set_xlabel('Configuration', fontsize=12)
    axes[0, 0].set_ylabel('CG X-position (mm)', fontsize=12)
    axes[0, 0].set_title('Center of Gravity Position', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Normal Forces
    axes[0, 1].plot(configs, n_fronts, 'go-', label='Front Wheel', linewidth=2, markersize=8)
    axes[0, 1].plot(configs, n_backs, 'mo-', label='Back Wheel', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    axes[0, 1].set_xlabel('Configuration', fontsize=12)
    axes[0, 1].set_ylabel('Normal Force (N)', fontsize=12)
    axes[0, 1].set_title('Normal Forces on Wheels', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Safety Margin (Front)
    safety_margin_front = n_fronts
    axes[1, 0].bar(configs, safety_margin_front, color=['green' if x > 0 else 'red' for x in safety_margin_front], 
                   alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Configuration', fontsize=12)
    axes[1, 0].set_ylabel('Front Wheel Safety Margin (N)', fontsize=12)
    axes[1, 0].set_title('Overturn Safety Margin', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Stability Index (percentage of weight on back)
    stability_index = n_backs / (n_fronts + n_backs) * 100
    axes[1, 1].plot(configs, stability_index, 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Configuration', fontsize=12)
    axes[1, 1].set_ylabel('Weight on Back Wheels (%)', fontsize=12)
    axes[1, 1].set_title('Weight Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_optimization_results(all_configs, all_times, all_overturned, best_config, best_time, save_path=None):
    """
    แสดงผลการ optimization
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Filter valid configurations
    valid_mask = ~all_overturned
    valid_configs = np.array(all_configs)[valid_mask]
    valid_times = np.array(all_times)[valid_mask]
    
    # Plot 1: All configurations
    ax1 = fig.add_subplot(131)
    colors = ['green' if not ot else 'red' for ot in all_overturned]
    ax1.scatter(range(len(all_configs)), all_times, c=colors, s=100, alpha=0.6, edgecolors='black')
    ax1.scatter(best_config['index'], best_time, c='gold', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Best Configuration', zorder=5)
    ax1.set_xlabel('Configuration Index', fontsize=12)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('All Tested Configurations', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of valid times
    ax2 = fig.add_subplot(132)
    ax2.hist(valid_times, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(best_time, color='red', linestyle='--', linewidth=2, label=f'Best: {best_time:.3f}s')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Valid Configurations', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Top 10 configurations
    ax3 = fig.add_subplot(133)
    sorted_indices = np.argsort(valid_times)[:10]
    top_times = valid_times[sorted_indices]
    top_labels = [f'Config {valid_configs[i]}' for i in sorted_indices]
    
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 10))
    bars = ax3.barh(range(10), top_times, color=colors_gradient, edgecolor='black')
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top_labels, fontsize=9)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_title('Top 10 Configurations', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, time) in enumerate(zip(bars, top_times)):
        ax3.text(time, i, f' {time:.3f}s', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sensitivity_analysis(param_name, param_values, results, baseline_value, save_path=None):
    """
    Sensitivity Analysis: ดูผลของการเปลี่ยนแปลงพารามิเตอร์
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute change
    axes[0].plot(param_values, results, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(baseline_value, color='r', linestyle='--', linewidth=2, label='Baseline')
    axes[0].set_xlabel(f'{param_name}', fontsize=12)
    axes[0].set_ylabel('Time (s)', fontsize=12)
    axes[0].set_title(f'Sensitivity to {param_name}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Percentage change
    baseline_result = results[np.argmin(np.abs(param_values - baseline_value))]
    percent_change = (results - baseline_result) / baseline_result * 100
    
    axes[1].plot(param_values, percent_change, 'ro-', linewidth=2, markersize=8)
    axes[1].axhline(0, color='k', linestyle='-', linewidth=1)
    axes[1].axvline(baseline_value, color='r', linestyle='--', linewidth=2, label='Baseline')
    axes[1].set_xlabel(f'{param_name}', fontsize=12)
    axes[1].set_ylabel('Change in Time (%)', fontsize=12)
    axes[1].set_title(f'Relative Sensitivity', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ==================== MAIN PROGRAM ====================

def main():
    # ---- Material Density ---- #
    SS400_DENSITY = 7.85e3
    ACRYLIC_DENSITY = 1.15e3

    # ---- Car Propoties ---- #
    CAR_VOLUME = 1.654e5 * 1e-9
    CAR_MASS = CAR_VOLUME * ACRYLIC_DENSITY

    CAR_HEIGHT = 98.4250085662*1e-3
    CAR_WIDTH = 135*1e-3

    CAR_CM_X = -2.808*1e-3
    CAR_CM_Y = CAR_HEIGHT - 35.573*1e-3

    RIGHT_WHEEL_X = 52.925*1e-3

    # ---- Rod Propoties ---- #
    ROD_POSITIONS = np.array([                       # มี 7 ตำแหน่ง
        [45, 98.425], 
        [30, 98.425], 
        [15, 98.425], 
        [0, 98.425], 
        [-15, 98.425], 
        [-30, 98.425], 
        [-45, 98.425]
        ]) * 1e-3
    
    ROD_LENGTHS = np.concatenate((
        [30]*2, [40]*2, [50]*2, [60]*2,
        [80]*2, [100]*2, [120]*2, [200]*2
    )) * 1e-3
    ROD_VOLUME = np.pi * (5*1e-3)**2 * ROD_LENGTHS
    ROD_MASSES = ROD_VOLUME * SS400_DENSITY           # มี 16 ลำดับตามไฟล์ excel

    # ---- Force Propoties ---- #
    PULLEY_POSITIONS = np.array([1500, 24.425]) * 1e-3

    FORCE_POSITIONS = np.array([[-60, 24.425], 
                              [-60, 30.425],
                              [-60, 36.425],
                              [-60, 42.425],
                              [-60, 48.425],
                              [-60, 54.425]]) * 1e-3
    PULLEY_LENGHTS = PULLEY_POSITIONS - FORCE_POSITIONS
    FORCE_ANGLE = np.arctan2(PULLEY_LENGHTS[:, 1], PULLEY_LENGHTS[:, 0])

    # ---- Parameter Selection ---- #
    force_rod_select = np.array([10, 11, 12, 13, 14, 15, 16])                                  # กรอกแท่งน้ำหนัก 1 - 16
    force_hole_select = 1                                 # เลือก 1-6 จาก ล่างขึ้นบน

    rod_select = np.array([7, 0, 0, 0, 0, 0, 0])         # กรอกแท่งน้ำหนัก ลำดับ 1-7 จากซ้ายไปขวา โดยเป็นเลขลำดับ 1 - 16

    Y0 = np.array([0.0, 0.0])  # [position, velocity]
    t_final = 1.0
    dt = 0.1

    x_max = None                                           # เลือกระยะทางเพื่อหาเวลา หรือ ถ้าต้องการหาระยะทางใส่ None
    
    # ---- Prepare for Calculation ---- #
    total_force = np.sum(ROD_MASSES[(force_rod_select - 1)]) * g
    force_position = FORCE_POSITIONS[force_hole_select-1]
    force_angle = np.abs(FORCE_ANGLE[force_hole_select-1])
    force = np.array([np.cos(force_angle), np.sin(force_angle)]) * total_force
    rod_mass = np.where(rod_select == 0, 0, ROD_MASSES[rod_select - 1])
    a = (force[0]) / (CAR_MASS + np.sum(rod_mass))

    n_steps = int(t_final / dt)

    # ---- Is the car overturned? ---- #
    cg = find_CG(CAR_CM_X, CAR_CM_Y, CAR_MASS, rod_mass, ROD_POSITIONS)
    n, is_overturned = find_N(cg[0], cg[1], CAR_MASS, RIGHT_WHEEL_X, ROD_POSITIONS, rod_mass, force, force_position)

    print(f"The car is {'overturned' if is_overturned else 'not overturned'} {n}")

    # ---- Numerical Method Test ---- #
    exact_result =  exact_solution(a, t = t_final, x_max=x_max)
    T_euler, Y_euler = euler_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps, x_max=x_max)
    T_improved, Y_improved = improved_euler_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps, x_max=x_max)
    T_rk4, Y_rk4 = rk4_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps, x_max=x_max)

    print(f"Exact:          X = {exact_result:.6f} m")
    print(f"Euler:          X = {Y_euler[-1, 0]:.6f} m")
    print(f"Improved Euler: X = {Y_improved[-1, 0]:.6f} m")
    print(f"RK4:            X = {Y_rk4[-1, 0]:.6f} m")

if __name__ == "__main__":
    main()


