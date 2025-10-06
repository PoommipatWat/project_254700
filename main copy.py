import numpy as np
from scipy.constants import g
    
def find_CG(CAR_CM_X, CAR_CM_Y, CAR_M, M_rod, rod_position):
    total_mass = CAR_M + np.sum(M_rod)

    sum_mx = CAR_CM_X * CAR_M + np.sum(M_rod * rod_position[:, 0])
    sum_my = CAR_CM_Y * CAR_M + np.sum(M_rod * rod_position[:, 1])

    CG_x = sum_mx / total_mass
    CG_y = sum_my / total_mass

    return [CG_x, CG_y, total_mass]

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
    moment_force = (force[0] * r_force_y) + (force[1] * r_force_x)

    r_N_x = 105.850/1000
    N_back = (moment_rod + moment_car - moment_force) / r_N_x
    N_front = (M_car + np.sum(M_rod))*g - force[1] - N_back

    return [N_front, N_back], N_front < 0 or N_back < 0

def main():

    Race_distance = 1 # Meter

    CAR_HEIGHT = 98.4250085662/1000 # Meter
    CAR_WIDTH = 135/1000 # Meter

    CAR_CM_X = -2.808/1000 # Meter
    CAR_CM_Y = CAR_HEIGHT - 35.573/1000 # Meter
    CAR_M = 1298.482/1000 # Kg

    WHEEL_X = 52.925/1000

    rod_position = np.array([
        [45, 98.425], 
        [30, 98.425], 
        [15, 98.425], 
        [0, 98.425], 
        [-15, 98.425], 
        [-30, 98.425], 
        [-45, 98.425]
    ]) * 0.001 # Meter
    M_rod = np.array([1,1,1,1,1,1,1]) # Kg

    force = 1  # N
    force_angle = 30
    force_matrix = np.array([np.cos(np.deg2rad(force_angle)), np.sin(np.deg2rad(force_angle))]) * force
    
    force_position = np.array([-62.5, 28.925]) * 0.001

    CG_total_mass = find_CG(CAR_CM_X, CAR_CM_Y, CAR_M, M_rod, rod_position)
    result2 = find_N(CG_total_mass[0], CG_total_mass[1], CAR_M, WHEEL_X, rod_position, M_rod, force_matrix, force_position)

    print(result2)
# เรียกใช้
if __name__ == "__main__":
    main()
