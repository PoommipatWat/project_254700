import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import time

# ==================== NUMERICAL METHODS ====================

def euler_method(f, x0, Y0, h, n_steps):
    """
    Euler Method (หน้า 5)
    Y(n+1) = Y(n) + h * f(x(n), Y(n))
    """
    x_values = np.zeros(n_steps + 1)
    Y_values = np.zeros((n_steps + 1, len(Y0)))
    
    x_values[0] = x0
    Y_values[0] = Y0
    
    for n in range(n_steps):
        x_n = x_values[n]
        Y_n = Y_values[n]
        
        Y_values[n + 1] = Y_n + h * f(x_n, Y_n)
        x_values[n + 1] = x_n + h
    
    return x_values, Y_values


def improved_euler_method(f, x0, Y0, h, n_steps):
    """
    Improved Euler Method / RK2 (หน้า 7)
    Predictor-Corrector
    """
    x_values = np.zeros(n_steps + 1)
    Y_values = np.zeros((n_steps + 1, len(Y0)))
    
    x_values[0] = x0
    Y_values[0] = Y0
    
    for n in range(n_steps):
        x_n = x_values[n]
        Y_n = Y_values[n]
        
        # Predictor
        Y_star = Y_n + h * f(x_n, Y_n)
        
        # Corrector
        Y_values[n + 1] = Y_n + (h / 2) * (f(x_n, Y_n) + f(x_n + h, Y_star))
        x_values[n + 1] = x_n + h
    
    return x_values, Y_values


def rk4_method(f, x0, Y0, h, n_steps):
    """
    Runge-Kutta 4th Order (หน้า 13)
    """
    x_values = np.zeros(n_steps + 1)
    Y_values = np.zeros((n_steps + 1, len(Y0)))
    
    x_values[0] = x0
    Y_values[0] = Y0
    
    for n in range(n_steps):
        x_n = x_values[n]
        Y_n = Y_values[n]
        
        k1 = h * f(x_n, Y_n)
        k2 = h * f(x_n + h/2, Y_n + k1/2)
        k3 = h * f(x_n + h/2, Y_n + k2/2)
        k4 = h * f(x_n + h, Y_n + k3)
        
        Y_values[n + 1] = Y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        x_values[n + 1] = x_n + h
    
    return x_values, Y_values


# ==================== STATE SPACE FUNCTION ====================

def F_state_space(t, Y, a):
    """
    State space function สำหรับความเร่งคงที่
    Y = [x, v]
    คืนค่า dY/dt = [v, a]
    
    Parameters:
    - t: เวลา
    - Y: state vector [x, v]
    - a: ความเร่งคงที่
    """
    x = Y[0]
    v = Y[1]
    
    dx_dt = v
    dv_dt = a
    
    return np.array([dx_dt, dv_dt])


def exact_solution(m1, m2, theta, t):
    """คำตอบแม่นตรง: x(t) = (1/2) * a * t²"""
    a = (m1 * g * np.cos(theta)) / m2
    return 0.5 * a * t**2


# ==================== MAIN PROGRAM ====================

def main():
    print("=" * 80)
    print("Numerical Methods สำหรับ ODE")
    print("=" * 80)
    
    # พารามิเตอร์
    m1 = 5.0
    m2 = 10.0
    theta = np.deg2rad(30)
    
    # คำนวณความเร่ง
    a = (m1 * g * np.cos(theta)) / m2
    
    # ค่าเริ่มต้น
    t0 = 0
    t_max = 10
    x0 = 0
    v0 = 0
    Y0 = np.array([x0, v0])
    
    # สร้าง function ที่มี 2 parameters โดยใช้ lambda
    # F(t, Y) จะเรียก F_state_space(t, Y, a) โดยส่ง a เข้าไปด้วย
    F = lambda t, Y: F_state_space(t, Y, a)
    
    print(f"\nสมการ: d²x/dt² = {a:.4f} m/s²")
    print(f"State Space: Y = [x, v]^T, dY/dt = [v, a]^T")
    print(f"ค่าเริ่มต้น: Y(0) = [{x0}, {v0}]^T\n")
    
    # ทดสอบหลาย dt
    dt_values = [2.0, 1.0, 0.5, 0.1]
    
    x_exact_final = exact_solution(m1, m2, theta, t_max)
    print(f"คำตอบแม่นตรง ที่ t = {t_max} s: x = {x_exact_final:.10f} m\n")
    
    print(f"{'dt':<8} {'Method':<15} {'x (m)':<18} {'v (m/s)':<18} {'Error (m)':<18} {'Steps':<8}")
    print("-" * 95)
    
    for dt in dt_values:
        n_steps = int(t_max / dt)
        x_exact = exact_solution(m1, m2, theta, t_max)
        
        # Euler
        t_e, Y_e = euler_method(F, t0, Y0, dt, n_steps)
        x_euler = Y_e[-1, 0]
        v_euler = Y_e[-1, 1]
        error_euler = abs(x_exact - x_euler)
        print(f"{dt:<8.2f} {'Euler':<15} {x_euler:<18.10f} {v_euler:<18.10f} {error_euler:<18.2e} {n_steps:<8}")
        
        # RK2
        t_rk2, Y_rk2 = improved_euler_method(F, t0, Y0, dt, n_steps)
        x_rk2 = Y_rk2[-1, 0]
        v_rk2 = Y_rk2[-1, 1]
        error_rk2 = abs(x_exact - x_rk2)
        print(f"{dt:<8.2f} {'RK2':<15} {x_rk2:<18.10f} {v_rk2:<18.10f} {error_rk2:<18.2e} {n_steps:<8}")
        
        # RK4
        t_rk4, Y_rk4 = rk4_method(F, t0, Y0, dt, n_steps)
        x_rk4 = Y_rk4[-1, 0]
        v_rk4 = Y_rk4[-1, 1]
        error_rk4 = abs(x_exact - x_rk4)
        print(f"{dt:<8.2f} {'RK4':<15} {x_rk4:<18.10f} {v_rk4:<18.10f} {error_rk4:<18.2e} {n_steps:<8}")
        
        print("-" * 95)
    
    # Plot
    dt_demo = 1.0
    n_steps = int(t_max / dt_demo)
    
    t_e, Y_e = euler_method(F, t0, Y0, dt_demo, n_steps)
    t_rk2, Y_rk2 = improved_euler_method(F, t0, Y0, dt_demo, n_steps)
    t_rk4, Y_rk4 = rk4_method(F, t0, Y0, dt_demo, n_steps)
    
    t_exact = np.linspace(0, t_max, 1000)
    x_exact_array = exact_solution(m1, m2, theta, t_exact)
    v_exact_array = a * t_exact
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Position
    ax1 = axes[0]
    ax1.plot(t_exact, x_exact_array, 'k-', label='Exact', linewidth=2)
    ax1.plot(t_e, Y_e[:, 0], 'o-', label='Euler', linewidth=2, markersize=6)
    ax1.plot(t_rk2, Y_rk2[:, 0], 's-', label='RK2', linewidth=2, markersize=6)
    ax1.plot(t_rk4, Y_rk4[:, 0], '^-', label='RK4', linewidth=2, markersize=6)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position x (m)', fontsize=12)
    ax1.set_title(f'Position vs Time (dt = {dt_demo} s)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Velocity
    ax2 = axes[1]
    ax2.plot(t_exact, v_exact_array, 'k-', label='Exact', linewidth=2)
    ax2.plot(t_e, Y_e[:, 1], 'o-', label='Euler', linewidth=2, markersize=6)
    ax2.plot(t_rk2, Y_rk2[:, 1], 's-', label='RK2', linewidth=2, markersize=6)
    ax2.plot(t_rk4, Y_rk4[:, 1], '^-', label='RK4', linewidth=2, markersize=6)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity v (m/s)', fontsize=12)
    ax2.set_title(f'Velocity vs Time (dt = {dt_demo} s)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("การใช้ lambda เพื่อส่ง parameter เพิ่มเติม:")
    print("F = lambda t, Y: F_state_space(t, Y, a)")
    print("=" * 80)


if __name__ == "__main__":
    main()