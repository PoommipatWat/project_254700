import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import time

# ==================== NUMERICAL METHODS ====================

def euler_method(f, x0, Y0, h, n_steps):
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
    x = Y[0]
    v = Y[1]
    
    dx_dt = v
    dv_dt = a
    
    return np.array([dx_dt, dv_dt])


# ==================== EXACT FUNCTION ====================
def exact_solution(a, t = None, x_max = None):
    if t is not None:
        return 0.5 * a * t**2 
    elif x_max is not None:
        return np.sqrt(2 * x_max * (1/a))
    

# ==================== MOMENT FUNCTION ====================
def find_CG(CAR_CM_X, CAR_CM_Y, CAR_M, M_rod, rod_position):
    total_mass = CAR_M + np.sum(M_rod)

    sum_mx = CAR_CM_X * CAR_M + np.sum(M_rod * rod_position[:, 0])
    sum_my = CAR_CM_Y * CAR_M + np.sum(M_rod * rod_position[:, 1])

    CG_x = sum_mx / total_mass
    CG_y = sum_my / total_mass

    return [CG_x, CG_y], total_mass

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


# ==================== MAIN PROGRAM ====================

def main():
    # ---- Material Density ---- #
    SS400_DENSITY = 7.85e3

    # ---- Car Propoties ---- #
    CAR_VOLUME = 1.654e5 * 1e-9
    CAR_MASS = CAR_VOLUME * SS400_DENSITY

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
    force_rod_select = 13                                  # กรอกแท่งน้ำหนัก 1 - 16
    force_hole_select = 1                                 # เลือก 1-6 จาก ล่างขึ้นบน

    rod_select = np.array([0, 0, 0, 5, 0, 0, 0])         # กรอกแท่งน้ำหนัก ลำดับ 1-7 จากซ้ายไปขวา โดยเป็นเลขลำดับ 1 - 16

    Y0 = np.array([0.0, 0.0])  # [position, velocity]
    t_final = 10.0
    dt = 0.01
    
    # ---- Prepare for Calculation ---- #
    total_force = ROD_MASSES[force_rod_select-1] * g
    force_position = FORCE_POSITIONS[force_hole_select-1]
    force_angle = FORCE_ANGLE[force_hole_select-1]
    force = np.array([np.cos(force_angle), np.sin(force_angle)]) * total_force
    rod_mass = np.where(rod_select == 0, 0, ROD_MASSES[rod_select - 1])
    a = (force[0]) / (CAR_MASS + np.sum(rod_mass))

    n_steps = int(t_final / dt)

    # ---- Is the car overturned? ---- #
    cg, total_mass = find_CG(CAR_CM_X, CAR_CM_Y, CAR_MASS, rod_mass, ROD_POSITIONS)
    n, is_overturned = find_N(cg[0], cg[1], CAR_MASS, RIGHT_WHEEL_X, ROD_POSITIONS, rod_mass, force, force_position)

    print(f"The car is {'overturned' if is_overturned else 'not overturned'}")

    # ---- Numerical Method Test ---- #
    exact_result =  exact_solution(a, t = t_final)
    T_euler, Y_euler = euler_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps)
    T_improved, Y_improved = improved_euler_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps)
    T_rk4, Y_rk4 = rk4_method(lambda t, Y: F_state_space(t, Y, a), 0, Y0, dt, n_steps)

    print(f"Exact:          X = {exact_result:.6f} m")
    print(f"Euler:          X = {Y_euler[-1, 0]:.6f} m")
    print(f"Improved Euler: X = {Y_improved[-1, 0]:.6f} m")
    print(f"RK4:            X = {Y_rk4[-1, 0]:.6f} m")

    print(exact_solution(a, x_max = exact_result))

if __name__ == "__main__":
    main()