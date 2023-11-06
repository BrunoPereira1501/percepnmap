import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def ang_normalized(ang):
    
    ang_normalized = ang

    while ang_normalized > np.pi:
        ang_normalized = ang_normalized - 2*np.pi
    
    while ang_normalized <= -np.pi:
        ang_normalized = ang_normalized + 2*np.pi

    return ang_normalized

# Iteration number
N = 2000

# Lists for the results 
X_t = []
X_e_t = []
Zmed = []
Zest = []
Ks = []

# Initialize an empty list to store the data
data_list = []

# Open the data file for reading
with open('Datasets-20231106/data1.txt', 'r') as file:
    for line in file:
        # Split the line into individual values using spaces as the delimiter
        values = line.split()
        
        # Convert the values to the appropriate data types (float in this case)
        t = float(values[0])
        x = float(values[1])
        y = float(values[2])
        theta = float(values[3])
        v = float(values[4])
        omega = float(values[5])
        r1 = float(values[6])
        psi1 = float(values[7])
        r2 = float(values[8])
        psi2 = float(values[9])
        
        # Create a dictionary for the current line and append it to the data list
        row_data = {
            't': t,
            'x': x,
            'y': y,
            'theta': theta,
            'v': v,
            'omega': omega,
            'r1': r1,
            'psi1': psi1,
            'r2': r2,
            'psi2': psi2
        }
        data_list.append(row_data)

# Get the data in the respective lists
time = [row['t'] for row in data_list]
x = [row['x'] for row in data_list]
y = [row['y'] for row in data_list]
theta = [row['theta'] for row in data_list]
v = [row['v'] for row in data_list]
w = [row['omega'] for row in data_list]
r1 = [row['r1'] for row in data_list]
psi1 = [row['psi1'] for row in data_list]
r2 = [row['r2'] for row in data_list]
psi2 = [row['psi2'] for row in data_list]


# Covariances
P = np.eye(3)*1e-3
Q = np.array([[0.5**2, 0],
     [0, 0.05**2 ]])

# Std for the beacons 
sdv_r = 0.5
sdv_psi = 0.1

R = np.array([[sdv_r ** 2, 0],
              [0, sdv_psi ** 2]])

# Variables to record simulation
X_t = []
X_e_t = []
Zmed = []
Zest = []
Ks = []

# Beacons coordinates
xp1 = 0
yp1 = 0
xp2 = 10
yp2 = 0

# initial value for the estimated state
X_e = [[x[0]], [y[0]], [theta[0]]]
X_e_t.append(X_e)
xr_e = x[0]
yr_e = y[0]
theta_r_e = theta[0]



# Sampling period
dt = 0.1

for i in range(int(N)):
    
    # Real Trajectory
    X = [[x[i]], [y[i]], [theta[i]]]
     
    xr_e = X_e_t[i][0][0]
    yr_e = X_e_t[i][1][0]
    theta_r_e = X_e_t[i][2][0]
    # Predict X(k+1) = f(X(k),U)
    x_k = xr_e + v[i]/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e)) 
    y_k = yr_e + v[i]/w[i] * (np.cos(theta_r_e) - np.cos(theta_r_e + w[i]*dt))
    theta_k = theta_r_e + w[i] * dt 
    X_e = np.array([[x_k], [y_k], [theta_k]])
    

    # Gradient of f(X)
    grad_f_X = np.array([[1, 0, v[i]/w[i] * (np.cos(theta_r_e + w[i]*dt) - np.cos(theta_r_e))],
            [0, 1, v[i]/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e))],
            [0, 0, 1]])
    
    # Gradient of f(U)
    grad_f_U = np.array([[1/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e)), v[i]/(w[i]**2) * (dt*w[i]*np.cos(theta_r_e + w[i]*dt) - np.sin(theta_r_e + w[i]*dt)+ np.sin(theta_r_e))],
            [1/w[i] * (-np.cos(theta_r_e + w[i]*dt) + np.cos(theta_r_e)), v[i]/(w[i]**2) * (dt*w[i]*np.sin(theta_r_e + w[i]*dt) + np.cos(theta_r_e + w[i]*dt) - np.cos(theta_r_e))],
            [0, dt]])

    # Covariance propagation
    P = grad_f_X @ P @ grad_f_X.T + grad_f_U @ Q @ grad_f_U.T

        
    for j in range(2):

        if j == 0:
            r, psi, landmark_x, landmark_y = r1[i], psi1[i], xp1, yp1
        else:
            r, psi, landmark_x, landmark_y = r2[i], psi2[i], xp2, yp2

        delta_x = landmark_x - X_e[0][0]
        delta_y = landmark_y - X_e[1][0]
        landmark_distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

        H = np.array([[landmark_distance],
                    [ang_normalized(math.atan2(delta_y, delta_x) - X_e[2][0])]])

        fH = np.array([[- delta_x / landmark_distance, - delta_y / landmark_distance, 0],
                    [delta_y / (landmark_distance ** 2), - delta_x / (landmark_distance ** 2), -1]])

        K = P @ fH.T @ np.linalg.inv(fH @ P @ fH.T + R)

        z = np.array([[r],
                    [psi]])

    
        z_dif = z - H
        z_dif[1] = ang_normalized(z_dif[1])
        X_e = X_e + K @ (z_dif)
        X_e[2] = ang_normalized(X_e[2])
        
        P = P - K @ (fH @ P @ fH.T + R) @ K.T
        # Save the results in lists 
    X_t.append(X) 
    X_e_t.append(X_e)
    Zmed.append(z) 
   

# Initialize empty lists for x and y values
x_real = []
y_real = []
x_est = []
y_est = []

# Extract x and y values from each array in the list
for array in X_t:
    x_real.append(array[0])  # Extract x (first element)
    y_real.append(array[1])  # Extract y (second element)

# Extract x and y values from each array in the list
for array in X_e_t:
    x_est.append(array[0])  # Extract x (first element)
    y_est.append(array[1])  # Extract y (second element)


print(X_e)
# # Create a function to update the plot in each animation frame
# def update(frame):
#     plt.clf()  # Clear the previous frame
#     plt.subplot(121)  # Subplot on the left
#     plt.scatter(x_real, y_real, label='Real Robot Position', color='b', s=5)
#     plt.scatter(x_est[:frame], y_est[:frame], label='Robot Position Estimation', color='r', s=5, linestyle='-')
#     plt.scatter(xp1, yp1, label='Beacon 1 Coordinates', color='yellow', marker='s')
#     plt.scatter(xp2, yp2, label='Beacon 2 Coordinates', color='orange', marker='s')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.title('Actual Robot Position Over Time')
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(122)  # Subplot on the right
#     plt.scatter(x_real[:frame], y_real[:frame], color='red', label='Real Trajectory', s=5)
#     plt.scatter(x_est[:frame], y_est[:frame], color='green', label='Estimated Trajectory', s=5)
#     plt.text(-5, 0, f'Frame: {frame}', fontsize=12, color='black')  # Add frame number as text
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()

# # Create a figure with two subplots
# plt.figure(figsize=(12, 6))

# # Create the initial plot
# ani = FuncAnimation(plt.gcf(), update, frames=len(x_real), repeat=False, interval=1)

# # Show the animation
# plt.show()

print(x_est)

# Create a plot for 'x' vs. 'y'
plt.figure(figsize=(8, 6))
plt.scatter(x_real, y_real, label='Real Robot Position', color='b', s = 5)
plt.scatter(x_est, y_est, label='Robot Position Estimation', color='r', s = 5)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Actual Robot Position Over Time')
plt.legend()
plt.grid(True)
plt.show()

