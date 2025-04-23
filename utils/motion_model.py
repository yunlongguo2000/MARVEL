"""
Compute the allowable heading for a drone given its current state and desired trajectory.

This function determines the achievable heading change based on the drone's current position,
final position, current heading, desired heading, current velocity, and maximum yaw rate.

Args:
    current_position (numpy.ndarray): Current (x, y) coordinates of the drone.
    final_position (numpy.ndarray): Target (x, y) coordinates of the drone.
    theta_current (float): Current heading angle in degrees.
    theta_desired (float): Desired heading angle in degrees.
    v_current (float): Current velocity of the drone.
    omega_max (float): Maximum yaw rate in degrees per second.

Returns:
    float: The achievable heading angle in the range [0, 360) degrees, 
           considering the drone's kinematic constraints.
"""
import math
import numpy as np

def normalize_angle(angle):
    """Normalize the angle to the range [0, 360) degrees."""
    return angle % 360

def compute_allowable_heading(current_position, final_position, theta_current, 
                           theta_desired, v_current, omega_max):
    
    x_current, y_current = current_position
    x_final, y_final = final_position

    # Calculate target heading based on current and final positions
    theta_target = math.degrees(math.atan2(y_final - y_current, x_final - x_current))
    theta_target = normalize_angle(theta_target)
    
    # Calculate the desired change in heading
    delta_theta_desired = normalize_angle(theta_desired) - normalize_angle(theta_current)
    
    # Normalize the desired change to the range [-180, 180]
    if delta_theta_desired > 180:
        delta_theta_desired -= 360
    elif delta_theta_desired < -180:
        delta_theta_desired += 360
    
    # Calculate time to achieve desired heading change
    t_desired_yaw = abs(delta_theta_desired) / omega_max
    
    # Calculate the distance to the final position
    distance_to_final = np.linalg.norm(final_position - current_position)
    
    # Calculate the time to reach the final position
    t_travel = distance_to_final / v_current
    
    # Check if the desired heading change is achievable within the travel time
    if t_desired_yaw <= t_travel:
        return normalize_angle(theta_desired)
    else:
        # Calculate the achievable heading change within the maximum yaw rate and travel time
        delta_theta_achievable = t_travel * omega_max
        theta_achievable = theta_current + math.copysign(delta_theta_achievable, delta_theta_desired)
        return normalize_angle(theta_achievable)

# # # Example usage:
# current_position = np.array([0, 0])
# final_position = np.array([1, 1])
# theta_current = 0  # in degrees
# theta_desired = 260  # in degrees
# v_current = 1  # in m/s
# omega_max = 35  # in degrees/s

# achievable_heading = compute_allowable_heading(current_position, final_position, theta_current, 
#                                             theta_desired, v_current, omega_max)
# print(achievable_heading)