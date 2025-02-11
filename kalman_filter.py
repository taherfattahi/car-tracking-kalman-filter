import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def kalman_filter(zs, dt, initial_state, initial_covariance, process_variance, measurement_variance):
    """
    Runs a 1D Kalman filter for tracking the relative state (distance and velocity)
    of the target vehicle with respect to the ego vehicle.
    
    Parameters:
      zs: list or array of LiDAR distance measurements (relative distance)
      dt: time step between measurements (seconds)
      initial_state: initial state vector (2x1 numpy array: [relative_distance, relative_velocity])
      initial_covariance: initial state covariance matrix (2x2 numpy array)
      process_variance: scalar used to build the process noise covariance Q
      measurement_variance: scalar representing the measurement noise variance R
      
    Returns:
      estimated_states: list of estimated state vectors (each is a 2x1 numpy array)
    """
    # Constant velocity model: state transition matrix
    F = np.array([[1, dt],
                  [0, 1]])
    
    # Measurement matrix: we only measure the relative distance
    H = np.array([[1, 0]])
    
    # Process noise covariance Q
    Q = process_variance * np.array([[dt**4 / 4, dt**3 / 2],
                                     [dt**3 / 2, dt**2]])
    
    # Measurement noise covariance R (1x1 matrix)
    R = np.array([[measurement_variance]])
    
    # Initialize state and covariance
    x = initial_state  # state vector: [relative_distance; relative_velocity]
    P = initial_covariance
    
    estimated_states = []
    
    for z in zs:
        # ----- Prediction Step -----
        x_pred = F.dot(x)
        P_pred = F.dot(P).dot(F.T) + Q
        
        # ----- Update Step -----
        z_meas = np.array([[z]])
        y = z_meas - H.dot(x_pred)    # Innovation (measurement residual)
        S = H.dot(P_pred).dot(H.T) + R # Innovation covariance
        K = P_pred.dot(H.T).dot(np.linalg.inv(S))  # Kalman gain
        
        x = x_pred + K.dot(y)         # Updated state estimate
        P = (np.eye(len(x)) - K.dot(H)).dot(P_pred) # Updated covariance
        
        estimated_states.append(x.copy())
        
    return estimated_states

# -------------------------
# Simulation and Animation Setup
# -------------------------
if __name__ == "__main__":
    # Simulation settings
    dt = 1.0         # Time step in seconds
    total_steps = 30 # Total number of simulation steps

    # ----- Ego Vehicle (Your Car) -----
    ego_initial_position = 0.0   # meters (starting position)
    ego_velocity = 22.0          # m/s (constant speed)

    # ----- Target Vehicle (Car in Front) -----
    target_initial_position = 50.0   # meters ahead initially
    target_velocity = 20.0           # m/s (constant speed)

    # Initial relative state (target relative to ego):
    # Relative distance = target_position - ego_position
    # Relative velocity = target_velocity - ego_velocity
    initial_relative_distance = target_initial_position - ego_initial_position
    initial_relative_velocity = target_velocity - ego_velocity
    initial_state = np.array([[initial_relative_distance], [initial_relative_velocity]])
    
    # Covariance and noise parameters
    initial_covariance = np.array([[1, 0],
                                   [0, 1]])
    process_variance = 0.1      # Process noise variance
    measurement_variance = 10.0 # LiDAR measurement noise variance

    # Prepare lists to store simulation data
    ego_positions = []             # Absolute positions of the ego car
    target_positions = []          # Absolute positions of the target car
    true_relative_distances = []   # True relative distances (target - ego)
    lidar_measurements = []        # Noisy LiDAR measurements (relative distance)

    ego_pos = ego_initial_position
    target_pos = target_initial_position

    np.random.seed(42)  # For reproducible noise
    for step in range(total_steps):
        # Update positions (assuming constant speeds)
        ego_pos += ego_velocity * dt
        target_pos += target_velocity * dt
        
        ego_positions.append(ego_pos)
        target_positions.append(target_pos)
        
        # True relative distance (target - ego)
        rel_distance = target_pos - ego_pos
        true_relative_distances.append(rel_distance)
        
        # Simulate LiDAR measurement with Gaussian noise
        noise = np.random.normal(0, np.sqrt(measurement_variance))
        meas = rel_distance + noise
        lidar_measurements.append(meas)

    # Run the Kalman filter on the LiDAR measurements
    estimates = kalman_filter(lidar_measurements, dt, initial_state, initial_covariance,
                              process_variance, measurement_variance)
    
    # Extract estimated relative distances and velocities
    estimated_relative_distances = [est[0, 0] for est in estimates]
    estimated_relative_velocities = [est[1, 0] for est in estimates]
    # Compute the estimated absolute target positions: ego position + estimated relative distance
    estimated_target_positions = [ego_positions[i] + estimated_relative_distances[i] for i in range(total_steps)]

    # -------------------------
    # Set Up the Visualization (Animated Road Scene)
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set static x-axis limits so that the movement is visible:
    # The ego car will move from 0 to (ego_velocity * total_steps)
    ax.set_xlim(0, ego_initial_position + ego_velocity * total_steps + 100)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Lane")
    ax.set_title("Ego & Target Car Tracking with LiDAR Measurements")
    
    # Draw a simple road center line
    ax.axhline(0, color='gray', lw=2, linestyle='--')
    
    # Car dimensions (in meters)
    ego_car_width, ego_car_height = 8, 2
    target_car_width, target_car_height = 8, 2
    
    # Create rectangular patches for the cars.
    # The rectangles are defined by their lower-left corner.
    ego_patch = patches.Rectangle((0, -ego_car_height/2), ego_car_width, ego_car_height,
                                  fc='blue', ec='black', label='Ego Car')
    target_patch = patches.Rectangle((0, -target_car_height/2), target_car_width, target_car_height,
                                     fc='red', ec='black', label='Target Car')
    # Patch for the Kalman filter's estimated target position.
    estimated_target_patch = patches.Rectangle((0, -target_car_height/2), target_car_width, target_car_height,
                                                 fc='none', ec='blue', linestyle='--',
                                                 linewidth=2, label='Estimated Target')
    
    # Add the patches to the axis
    ax.add_patch(ego_patch)
    ax.add_patch(target_patch)
    ax.add_patch(estimated_target_patch)
    
    # Draw a line representing the LiDAR measurement (from the ego car's center to the target car's center)
    lidar_line, = ax.plot([], [], 'g--', lw=2, label="LiDAR Measurement")
    
    # Text annotation to display the estimated relative state
    est_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Display the legend
    ax.legend(loc='upper left')
    ax.grid(True)

    # -------------------------
    # Animation Update Function
    # -------------------------
    def update(frame):
        # Get the current positions
        ego_x = ego_positions[frame]
        target_x = target_positions[frame]
        estimated_target_x = estimated_target_positions[frame]
        
        # Update the ego car patch position
        ego_patch.set_xy((ego_x - ego_car_width/2, -ego_car_height/2))
        # Update the target car patch position
        target_patch.set_xy((target_x - target_car_width/2, -target_car_height/2))
        # Update the estimated target patch position
        estimated_target_patch.set_xy((estimated_target_x - target_car_width/2, -target_car_height/2))
        
        # Update the LiDAR measurement line (from ego center to target center)
        lidar_line.set_data([ego_x, target_x], [0, 0])
        
        # Update the text annotation with current estimated relative state
        est_text.set_text(
            f"Estimated Relative Distance: {estimated_relative_distances[frame]:.2f} m\n"
            f"Estimated Relative Velocity: {estimated_relative_velocities[frame]:.2f} m/s"
        )
        
        return ego_patch, target_patch, estimated_target_patch, lidar_line, est_text

    # Create the animation
    anim = FuncAnimation(fig, update, frames=total_steps, interval=500, blit=True, repeat=False)

    plt.show()

    # Optionally, print the estimated states (relative distance and velocity) for each time step.
    for i, est in enumerate(estimates):
        print(f"Time {i+1:2d}: Relative Distance = {est[0, 0]:6.2f} m, Relative Velocity = {est[1, 0]:6.2f} m/s")
