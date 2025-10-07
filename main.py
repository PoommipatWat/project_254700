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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

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
    # แปลงเป็น numpy array
    step_sizes = np.array(step_sizes)
    errors_euler = np.array(errors_euler)
    errors_improved = np.array(errors_improved)
    errors_rk4 = np.array(errors_rk4)
    
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
    valid_mask = ~np.array(all_overturned)
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


import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import time
import itertools

# สมมติว่า import functions จากไฟล์อื่น
# from trolley_race import euler_method, improved_euler_method, rk4_method, F_state_space, exact_solution
# from trolley_race import find_CG, find_N
# from plotting_functions import *

import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import time
import itertools

# สมมติว่า import functions จากไฟล์อื่น
# from trolley_race import euler_method, improved_euler_method, rk4_method, F_state_space, exact_solution
# from trolley_race import find_CG, find_N
# from plotting_functions import *

# ==================== HELPER FUNCTIONS ====================

def calculate_system(force_rod_select, force_hole_select, rod_select, 
                     CAR_MASS, ROD_MASSES, ROD_POSITIONS, FORCE_POSITIONS, 
                     PULLEY_POSITIONS, FORCE_ANGLE, CAR_CM_X, CAR_CM_Y, RIGHT_WHEEL_X):
    """
    คำนวณคุณสมบัติของระบบสำหรับ configuration ที่กำหนด
    """
    # คำนวณแรง
    total_force = np.sum(ROD_MASSES[(force_rod_select - 1)]) * g
    force_position = FORCE_POSITIONS[force_hole_select-1]
    force_angle = np.abs(FORCE_ANGLE[force_hole_select-1])
    force = np.array([np.cos(force_angle), np.sin(force_angle)]) * total_force
    
    # คำนวณมวลรวม
    rod_mass = np.where(rod_select == 0, 0, ROD_MASSES[rod_select - 1])
    total_mass = CAR_MASS + np.sum(rod_mass)
    
    # คำนวณความเร่ง
    a = force[0] / total_mass
    
    # หา CG และเช็คการคว่ำ
    cg = find_CG(CAR_CM_X, CAR_CM_Y, CAR_MASS, rod_mass, ROD_POSITIONS)
    n, is_overturned = find_N(cg[0], cg[1], CAR_MASS, RIGHT_WHEEL_X, 
                              ROD_POSITIONS, rod_mass, force, force_position)
    
    return {
        'acceleration': a,
        'force': force,
        'force_angle': force_angle,
        'cg': cg,
        'normal_forces': n,
        'is_overturned': is_overturned,
        'total_mass': total_mass,
        'rod_mass': rod_mass
    }


def run_simulation(a, method, dt, x_max, Y0):
    """
    รันการจำลองด้วย method ที่กำหนด
    """
    n_steps = int(10.0 / dt)  # เพียงพอสำหรับการวิ่ง
    
    if method == 'euler':
        T, Y = euler_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps, x_max=x_max)
    elif method == 'improved':
        T, Y = improved_euler_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps, x_max=x_max)
    elif method == 'rk4':
        T, Y = rk4_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps, x_max=x_max)
    
    return T, Y


# ==================== EXPERIMENT 1: NUMERICAL METHODS ====================

def experiment_1_accuracy_test(a, x_max, dt, Y0, save_plots=False):
    """
    การทดลองที่ 1.1: ทดสอบความแม่นยำของแต่ละวิธี
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.1: ACCURACY TEST")
    print("="*60)
    
    # รันแต่ละวิธี
    T_euler, Y_euler = run_simulation(a, 'euler', dt, x_max, Y0)
    T_improved, Y_improved = run_simulation(a, 'improved', dt, x_max, Y0)
    T_rk4, Y_rk4 = run_simulation(a, 'rk4', dt, x_max, Y0)
    
    # คำนวณ exact solution
    t_exact = exact_solution(a, x_max=x_max)
    x_exact = x_max
    
    # คำนวณ error
    error_euler = abs(Y_euler[-1, 0] - x_exact)
    error_improved = abs(Y_improved[-1, 0] - x_exact)
    error_rk4 = abs(Y_rk4[-1, 0] - x_exact)
    
    # แสดงผล
    print(f"\nTarget distance: {x_max} m")
    print(f"Time step: {dt} s")
    print(f"\nExact time:          {t_exact:.6f} s")
    print(f"Euler time:          {T_euler[-1]:.6f} s (Error: {error_euler:.6e} m)")
    print(f"Improved Euler time: {T_improved[-1]:.6f} s (Error: {error_improved:.6e} m)")
    print(f"RK4 time:            {T_rk4[-1]:.6f} s (Error: {error_rk4:.6e} m)")
    
    # พล็อตกราฟ
    plot_trajectory_comparison(T_euler, Y_euler, T_improved, Y_improved, T_rk4, Y_rk4, 
                              exact_solution, a, 
                              save_path='exp1.1_trajectory.png' if save_plots else None)
    
    return {
        'euler': {'time': T_euler[-1], 'error': error_euler},
        'improved': {'time': T_improved[-1], 'error': error_improved},
        'rk4': {'time': T_rk4[-1], 'error': error_rk4}
    }


def experiment_2_convergence_test(a, x_max, Y0, save_plots=False):
    """
    การทดลองที่ 1.2: ทดสอบผลของ Step Size
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.2: CONVERGENCE TEST")
    print("="*60)
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    errors_euler = []
    errors_improved = []
    errors_rk4 = []
    
    # คำนวณเวลาที่แน่นอนสำหรับระยะ x_max
    t_final = exact_solution(a, x_max=x_max)
    x_exact = x_max
    
    for dt in step_sizes:
        print(f"\nTesting dt = {dt} s...")
        
        # รันแต่ละวิธีจนถึงเวลา t_final (ไม่ใช้ x_max)
        T_euler, Y_euler = run_simulation(a, 'euler', dt, None, Y0)
        T_improved, Y_improved = run_simulation(a, 'improved', dt, None, Y0)
        T_rk4, Y_rk4 = run_simulation(a, 'rk4', dt, None, Y0)
        
        # หาค่า position ที่เวลา t_final (โดย interpolation หรือหาจุดใกล้เคียง)
        idx_euler = np.argmin(np.abs(T_euler - t_final))
        idx_improved = np.argmin(np.abs(T_improved - t_final))
        idx_rk4 = np.argmin(np.abs(T_rk4 - t_final))
        
        # คำนวณ error ที่เวลาเดียวกัน
        errors_euler.append(abs(Y_euler[idx_euler, 0] - x_exact))
        errors_improved.append(abs(Y_improved[idx_improved, 0] - x_exact))
        errors_rk4.append(abs(Y_rk4[idx_rk4, 0] - x_exact))
        
        print(f"  Euler error: {errors_euler[-1]:.6e} m")
        print(f"  Improved Euler error: {errors_improved[-1]:.6e} m")
        print(f"  RK4 error: {errors_rk4[-1]:.6e} m")
    
    # พล็อตกราฟ
    plot_error_analysis(step_sizes, errors_euler, errors_improved, errors_rk4,
                       save_path='exp1.2_convergence.png' if save_plots else None)
    
    # คำนวณ order of convergence
    print("\n" + "-"*60)
    print("Order of Convergence (approximate):")
    for i in range(len(step_sizes)-1):
        ratio = step_sizes[i] / step_sizes[i+1]
        order_euler = np.log(errors_euler[i] / errors_euler[i+1]) / np.log(ratio)
        order_improved = np.log(errors_improved[i] / errors_improved[i+1]) / np.log(ratio)
        order_rk4 = np.log(errors_rk4[i] / errors_rk4[i+1]) / np.log(ratio)
        
        print(f"\ndt: {step_sizes[i]} -> {step_sizes[i+1]}")
        print(f"  Euler: {order_euler:.2f} (expected: 1)")
        print(f"  Improved Euler: {order_improved:.2f} (expected: 2)")
        print(f"  RK4: {order_rk4:.2f} (expected: 4)")
    
    return step_sizes, errors_euler, errors_improved, errors_rk4


def experiment_3_efficiency_test(a, x_max, Y0, save_plots=False):
    """
    การทดลองที่ 1.3: ทดสอบ Computational Efficiency
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.3: COMPUTATIONAL EFFICIENCY TEST")
    print("="*60)
    
    dt = 0.001  # step size เล็กเพื่อให้เห็นความแตกต่างของเวลา
    methods = ['Euler', 'Improved Euler', 'RK4']
    times = []
    errors = []
    
    x_exact = x_max
    
    for method_name, method_func in [('Euler', 'euler'), 
                                      ('Improved Euler', 'improved'), 
                                      ('RK4', 'rk4')]:
        print(f"\nTesting {method_name}...")
        
        start = time.time()
        _, Y = run_simulation(a, method_func, dt, x_max, Y0)
        end = time.time()
        
        comp_time = end - start
        error = abs(Y[-1, 0] - x_exact)
        
        times.append(comp_time)
        errors.append(error)
        
        print(f"  Computation time: {comp_time:.6f} s")
        print(f"  Error: {error:.6e} m")
    
    # พล็อตกราฟ
    plot_efficiency_comparison(methods, times, errors,
                              save_path='exp1.3_efficiency.png' if save_plots else None)
    
    return methods, times, errors


# ==================== EXPERIMENT 2: CONFIGURATION OPTIMIZATION ====================

def experiment_4_force_hole_study(force_rod_select, rod_select, config_params, save_plots=False):
    """
    การทดลองที่ 2.1A: ศึกษาผลของตำแหน่งเชือก
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.1A: FORCE HOLE POSITION STUDY")
    print("="*60)
    
    hole_positions = range(1, 7)
    results = []
    
    for hole in hole_positions:
        print(f"\nTesting hole position {hole}...")
        
        system = calculate_system(force_rod_select, hole, rod_select, **config_params)
        
        if system['is_overturned']:
            print(f"  -> OVERTURNED!")
            results.append({
                'hole': hole,
                'time': np.inf,
                'acceleration': system['acceleration'],
                'force': system['force'],
                'force_angle': system['force_angle'],
                'overturned': True
            })
        else:
            # คำนวณเวลา
            x_max = 1.5
            t_final = exact_solution(system['acceleration'], x_max=x_max)
            
            print(f"  Acceleration: {system['acceleration']:.3f} m/s²")
            print(f"  Time: {t_final:.3f} s")
            print(f"  Force angle: {np.degrees(system['force_angle']):.2f}°")
            
            results.append({
                'hole': hole,
                'time': t_final,
                'acceleration': system['acceleration'],
                'force': system['force'],
                'force_angle': system['force_angle'],
                'overturned': False
            })
    
    # สร้างกราฟ
    angles = np.array([r['force_angle'] for r in results])
    forces = np.array([r['force'] for r in results])
    accelerations = np.array([r['acceleration'] for r in results])
    
    plot_force_angle_analysis(list(hole_positions), angles, forces, accelerations,
                             save_path='exp2.1a_force_hole.png' if save_plots else None)
    
    return results


def experiment_5_mass_distribution_study(force_rod_select, force_hole_select, config_params, save_plots=False):
    """
    การทดลองที่ 2.1B: ศึกษาผลของการกระจายมวล
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.1B: MASS DISTRIBUTION STUDY")
    print("="*60)
    
    print(f"Using force rods: {force_rod_select}")
    
    # สร้าง test cases
    test_cases = {
        'All on platform': np.array([0, 0, 0, 0, 0, 0, 0]),
        'Front heavy': np.array([1, 2, 3, 4, 0, 0, 0]),
        'Back heavy': np.array([0, 0, 0, 5, 6, 7, 8]),
        'Balanced': np.array([1, 0, 3, 0, 5, 0, 7]),
        'Center concentrated': np.array([0, 0, 0, 1, 0, 0, 0])
    }
    
    results = []
    
    for name, rod_config in test_cases.items():
        print(f"\nTesting '{name}' configuration...")
        
        system = calculate_system(force_rod_select, force_hole_select, rod_config, **config_params)
        
        if system['is_overturned']:
            print(f"  -> OVERTURNED!")
            results.append({
                'name': name,
                'time': np.inf,
                'cg': system['cg'],
                'n_front': system['normal_forces'][0],
                'n_back': system['normal_forces'][1],
                'overturned': True
            })
        else:
            x_max = 1.5
            t_final = exact_solution(system['acceleration'], x_max=x_max)
            
            print(f"  Time: {t_final:.3f} s")
            print(f"  CG position: ({system['cg'][0]*1000:.2f}, {system['cg'][1]*1000:.2f}) mm")
            print(f"  Normal forces: Front={system['normal_forces'][0]:.2f} N, Back={system['normal_forces'][1]:.2f} N")
            
            results.append({
                'name': name,
                'time': t_final,
                'cg': system['cg'],
                'n_front': system['normal_forces'][0],
                'n_back': system['normal_forces'][1],
                'overturned': False
            })
    
    # สร้างกราฟ
    configs = [r['name'] for r in results]
    cg_positions = np.array([r['cg'] for r in results])
    n_fronts = np.array([r['n_front'] for r in results])
    n_backs = np.array([r['n_back'] for r in results])
    
    plot_stability_analysis(configs, cg_positions, n_fronts, n_backs,
                           save_path='exp2.1b_mass_distribution.png' if save_plots else None)
    
    return results


def experiment_6_optimization(config_params, x_max=1.5, save_plots=False):
    """
    การทดลองที่ 2.2: หา Configuration ที่ดีที่สุด
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.2: CONFIGURATION OPTIMIZATION")
    print("="*60)
    print("\nSearching for optimal configuration...")
    print("This may take a few minutes...\n")
    
    # ลองใช้มวลที่หลากหลาย ไม่ใช่หนักสุดอย่างเดียว
    force_rod_options = [
        np.array([13, 14, 15, 16]),        # 4 แท่งหนักสุด
        np.array([10, 11, 12, 13, 14]),    # 5 แท่งหนัก
        np.array([9, 10, 11, 12, 13, 14]), # 6 แท่งกลาง-หนัก
    ]
    
    best_time = np.inf
    best_config = None
    all_results = []
    
    # ทดสอบทุกตำแหน่งของเชือก
    for hole in range(1, 7):
        print(f"Testing hole position {hole}...")
        
        # ทดสอบหลายแบบของการวางมวลบนรถ (sampling)
        # เนื่องจากมีความเป็นไปได้มาก ให้ทดสอบแค่บางกรณี
        
        # Strategy 1: ทั้งหมดอยู่บนแท่น (เพื่อแรงดึงสูงสุด)
        rod_select = np.array([0, 0, 0, 0, 0, 0, 0])
        system = calculate_system(force_rod_options[0], hole, rod_select, **config_params)
        
        if not system['is_overturned']:
            t_final = exact_solution(system['acceleration'], x_max=x_max)
            all_results.append({
                'hole': hole,
                'rod_config': rod_select.copy(),
                'time': t_final,
                'overturned': False
            })
            
            if t_final < best_time:
                best_time = t_final
                best_config = {
                    'hole': hole,
                    'rod_config': rod_select.copy(),
                    'index': len(all_results) - 1
                }
                print(f"  -> New best! Time: {t_final:.4f} s")
        else:
            all_results.append({
                'hole': hole,
                'rod_config': rod_select.copy(),
                'time': np.inf,
                'overturned': True
            })
        
        # Strategy 2-5: ลองวางมวลบนรถในตำแหน่งต่างๆ (เพื่อป้องกันคว่ำ)
        test_positions = [
            np.array([1, 0, 0, 0, 0, 0, 0]),  # ด้านหน้าสุด
            np.array([0, 0, 0, 1, 0, 0, 0]),  # กึ่งกลาง
            np.array([0, 0, 0, 0, 0, 0, 1]),  # ด้านหลังสุด
            np.array([1, 0, 2, 0, 0, 0, 0]),  # หลายตำแหน่ง
        ]
        
        for rod_select in test_positions:
            system = calculate_system(force_rod_options[0], hole, rod_select, **config_params)
            
            if not system['is_overturned']:
                t_final = exact_solution(system['acceleration'], x_max=x_max)
                all_results.append({
                    'hole': hole,
                    'rod_config': rod_select.copy(),
                    'time': t_final,
                    'overturned': False
                })
                
                if t_final < best_time:
                    best_time = t_final
                    best_config = {
                        'hole': hole,
                        'rod_config': rod_select.copy(),
                        'index': len(all_results) - 1
                    }
                    print(f"  -> New best! Time: {t_final:.4f} s")
            else:
                all_results.append({
                    'hole': hole,
                    'rod_config': rod_select.copy(),
                    'time': np.inf,
                    'overturned': True
                })
    
    # แสดงผลลัพท์
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    if best_config is None:
        print("\n❌ ERROR: No valid configuration found!")
        print("   All tested configurations resulted in overturn.")
        print("   Suggestion: Try using lighter weights or different positions.")
        
        # สร้างกราฟแม้ไม่มี best config
        all_times = [r['time'] for r in all_results]
        all_overturned = [r['overturned'] for r in all_results]
        
        print(f"\n   Total configurations tested: {len(all_results)}")
        print(f"   All overturned: {sum(all_overturned)}")
        
        # Return dummy values
        return None, np.inf, all_results
    
    # print(f"\n✓ Best configuration found:")
    # print(f"  Force rods used: {best_config['force_rods']}")
    # print(f"  Hole position: {best_config['hole']}")
    # print(f"  Rod configuration: {best_config['rod_config']}")
    # print(f"  Time: {best_time:.4f} s")
    
    # พล็อตผลลัพท์
    all_times = [r['time'] for r in all_results]
    all_overturned = [r['overturned'] for r in all_results]
    
    # if best_config is not None and save_plots:
    #     plot_optimization_results(range(len(all_results)), all_times, all_overturned, 
    #                              best_config, best_time,
    #                              save_path='exp2.2_optimization.png')
    
    return best_config, best_time, all_results


def experiment_7_sensitivity_analysis(best_config, config_params, x_max=1.5, save_plots=False):
    """
    การทดลองที่ 3.1: Sensitivity Analysis
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3.1: SENSITIVITY ANALYSIS")
    print("="*60)
    
    if best_config is None:
        print("\n⚠ Skipping sensitivity analysis (no valid configuration found)")
        return [], []
    
    force_rod_select = best_config.get('force_rods', np.array([10, 11, 12, 13]))
    
    # Test 1: Sensitivity to CG position
    print("\nTesting sensitivity to CG position...")
    
    baseline_system = calculate_system(force_rod_select, best_config['hole'], 
                                      best_config['rod_config'], **config_params)
    baseline_time = exact_solution(baseline_system['acceleration'], x_max=x_max)
    
    # Vary rod position (simulate CG change)
    rod_positions_test = []
    times_test = []
    
    for i in range(7):
        rod_test = np.zeros(7, dtype=int)
        rod_test[i] = 1  # วางแท่งเดียวในตำแหน่งต่างๆ
        
        system = calculate_system(force_rod_select, best_config['hole'], 
                                 rod_test, **config_params)
        
        if not system['is_overturned']:
            t = exact_solution(system['acceleration'], x_max=x_max)
            rod_positions_test.append(i)
            times_test.append(t)
    
    # พล็อต
    # if len(rod_positions_test) > 0:
    #     plot_sensitivity_analysis('Rod Position', rod_positions_test, times_test, 3,
    #                              save_path='exp3.1_sensitivity.png' if save_plots else None)
    
    return rod_positions_test, times_test


# ==================== MAIN EXECUTION ====================

def main():
    """
    รัน experiments ทั้งหมด
    """
    print("\n" + "="*80)
    print(" "*20 + "TROLLEY RACE EXPERIMENTS")
    print("="*80)
    
    # ตั้งค่าพารามิเตอร์
    SS400_DENSITY = 7.85e3
    ACRYLIC_DENSITY = 1.15e3
    
    CAR_VOLUME = 1.654e5 * 1e-9
    CAR_MASS = CAR_VOLUME * ACRYLIC_DENSITY
    CAR_HEIGHT = 98.4250085662*1e-3
    CAR_CM_X = -2.808*1e-3
    CAR_CM_Y = CAR_HEIGHT - 35.573*1e-3
    RIGHT_WHEEL_X = 52.925*1e-3
    
    ROD_POSITIONS = np.array([
        [45, 98.425], [30, 98.425], [15, 98.425], [0, 98.425],
        [-15, 98.425], [-30, 98.425], [-45, 98.425]
    ]) * 1e-3
    
    ROD_LENGTHS = np.concatenate((
        [30]*2, [40]*2, [50]*2, [60]*2,
        [80]*2, [100]*2, [120]*2, [200]*2
    )) * 1e-3
    ROD_VOLUME = np.pi * (5*1e-3)**2 * ROD_LENGTHS
    ROD_MASSES = ROD_VOLUME * SS400_DENSITY
    
    PULLEY_POSITIONS = np.array([1500, 24.425]) * 1e-3
    FORCE_POSITIONS = np.array([
        [-60, 24.425], [-60, 30.425], [-60, 36.425],
        [-60, 42.425], [-60, 48.425], [-60, 54.425]
    ]) * 1e-3
    PULLEY_LENGHTS = PULLEY_POSITIONS - FORCE_POSITIONS
    FORCE_ANGLE = np.arctan2(PULLEY_LENGHTS[:, 1], PULLEY_LENGHTS[:, 0])
    
    config_params = {
        'CAR_MASS': CAR_MASS,
        'ROD_MASSES': ROD_MASSES,
        'ROD_POSITIONS': ROD_POSITIONS,
        'FORCE_POSITIONS': FORCE_POSITIONS,
        'PULLEY_POSITIONS': PULLEY_POSITIONS,
        'FORCE_ANGLE': FORCE_ANGLE,
        'CAR_CM_X': CAR_CM_X,
        'CAR_CM_Y': CAR_CM_Y,
        'RIGHT_WHEEL_X': RIGHT_WHEEL_X
    }
    
    # ค่าเริ่มต้นสำหรับการทดสอบ
    Y0 = np.array([0.0, 0.0])
    x_max = 1.5  # ระยะทาง 1.5 m
    dt = 0.01
    
    # สำหรับ numerical methods test - ใช้ configuration ที่ไม่คว่ำ
    force_rod_select = np.array([10, 11, 12, 13, 14, 15, 16])
    rod_select = np.array([0, 0, 0, 0, 0, 0, 0])  # ทั้งหมดบนแท่น
    force_hole_select = 3
    
    system = calculate_system(force_rod_select, force_hole_select, rod_select, **config_params)
    a = system['acceleration']
    
    # ==================== RUN EXPERIMENTS ====================
    
    # Part 1: Numerical Methods Analysis
    print("\n" + "#"*80)
    print(" "*20 + "PART 1: NUMERICAL METHODS ANALYSIS")
    print("#"*80)
    
    results_1_1 = experiment_1_accuracy_test(a, x_max, dt, Y0, save_plots=True)
    results_1_2 = experiment_2_convergence_test(a, x_max, Y0, save_plots=True)
    results_1_3 = experiment_3_efficiency_test(a, x_max, Y0, save_plots=True)
    
    # Part 2: Configuration Optimization
    print("\n" + "#"*80)
    print(" "*20 + "PART 2: CONFIGURATION OPTIMIZATION")
    print("#"*80)
    
    results_2_1a = experiment_4_force_hole_study(force_rod_select, rod_select, 
                                                 config_params, save_plots=True)
    results_2_1b = experiment_5_mass_distribution_study(force_rod_select,
                                                        force_hole_select, 
                                                        config_params, save_plots=True)
    best_config, best_time, optimization_results = experiment_6_optimization(config_params, 
                                                                     x_max, save_plots=True)
    
    # Part 3: Advanced Analysis
    print("\n" + "#"*80)
    print(" "*20 + "PART 3: ADVANCED ANALYSIS")
    print("#"*80)
    
    if best_config is not None:
        results_3_1 = experiment_7_sensitivity_analysis(best_config, config_params, 
                                                        x_max, save_plots=True)
    else:
        print("\n⚠ Skipping Part 3 (no valid configuration found)")
        results_3_1 = ([], [])
    
    # ==================== FINAL SUMMARY ====================
    
    print("\n" + "="*80)
    print(" "*30 + "FINAL SUMMARY")
    print("="*80)
    
    print("\n1. NUMERICAL METHODS:")
    print(f"   - Most accurate: RK4 (Error: {results_1_1['rk4']['error']:.2e} m)")
    print(f"   - Fastest: Euler (but least accurate)")
    print(f"   - Best balance: RK4 (high accuracy, acceptable speed)")
    
    print("\n2. OPTIMAL CONFIGURATION:")
    if best_config is not None:
        print(f"   - Best time achieved: {best_time:.4f} s")
        print(f"   - Force rods: {best_config.get('force_rods', 'N/A')}")
        print(f"   - Hole position: {best_config['hole']}")
        print(f"   - Rod configuration: {best_config['rod_config']}")
    else:
        print(f"   ❌ No valid configuration found!")
        print(f"   - All tested configurations resulted in overturn")
        print(f"   - Consider: using fewer/lighter weights or adjusting positions")
    
    print("\n3. KEY FINDINGS:")
    if len(optimization_results) > 0:
        valid_count = sum([1 for r in optimization_results if not r['overturned']])
        overturn_count = sum([1 for r in optimization_results if r['overturned']])
        print(f"   - Total configurations tested: {len(optimization_results)}")
        print(f"   - Valid configurations: {valid_count}")
        print(f"   - Overturned configurations: {overturn_count}")
        if valid_count > 0:
            valid_times = [r['time'] for r in optimization_results if not r['overturned']]
            print(f"   - Time range: {min(valid_times):.4f} - {max(valid_times):.4f} s")
    else:
        print(f"   - No results to analyze")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # ต้อง import functions จากไฟล์เดิม
    # from original_code import euler_method, improved_euler_method, rk4_method
    # from original_code import F_state_space, exact_solution, find_CG, find_N
    # from plotting_functions import *
    
    main()