import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
with open('Datasets-20231106/data3.txt', 'r') as file:
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
        r = float(values[6])
        psi = float(values[7])
    
        # Create a dictionary for the current line and append it to the data list
        row_data = {
            't': t,
            'x': x,
            'y': y,
            'theta': theta,
            'v': v,
            'omega': omega,
            'r': r,
            'psi': psi
            }
        data_list.append(row_data)

# Get the data in the respective lists
time = [row['t'] for row in data_list]
x = [row['x'] for row in data_list]
y = [row['y'] for row in data_list]
theta = [row['theta'] for row in data_list]
v = [row['v'] for row in data_list]
w = [row['omega'] for row in data_list]
r = [row['r'] for row in data_list]
psi = [row['psi'] for row in data_list]



# Covariances
P = np.eye(3)*1e-2
Q = np.array([[0.5**2, 0],
     [0, 0.05**2 ]])

# Std for the beacons 
sdv_r = 0.5
sdv_psi = 0.1

# Variables to record simulation
X_t = []
X_e_t = []
Zmed = []
Zest = []
Ks = []
B = []

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
    xk= xr_e + v[i]/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e)) 
    yk = yr_e + v[i]/w[i] * (np.cos(theta_r_e) - np.cos(theta_r_e + w[i]*dt))
    thetak = theta_r_e + w[i] * dt 
    X_e = np.array([[xk], [yk], [thetak]])

    # Gradient of f(X)
    grad_f_X = np.array([[1, 0, v[i]/w[i] * (np.cos(theta_r_e + w[i]*dt) - np.cos(theta_r_e))],
            [0, 1, v[i]/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e))],
            [0, 0, 1]])
    
    # Gradient of f(U)
    grad_f_U = np.array([[1/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e)), v[i]/(w[i]**2) * (dt*w[i]*np.cos(theta_r_e + w[i]*dt) - np.sin(theta_r_e + w[i]*dt)+ np.sin(theta_r_e))],
            [1/w[i] * (-np.cos(theta_r_e + w[i]*dt) + np.cos(theta_r_e)), v[i]/(w[i]**2) * (dt*w[i]*np.sin(theta_r_e + w[i]*dt) + np.cos(theta_r_e + w[i]*dt) - np.cos(theta_r_e))],
            [0, dt]])

    # Covariance propagation
    P = grad_f_X @ P @ grad_f_X.transpose() + grad_f_U @ Q @ grad_f_U.transpose()

    landmark_loc = np.array([[X_e[0][0] + r[i] * np.cos(psi[i] + X_e[2][0])],
                                 [X_e[1][0] + r[i] * np.sin(psi[i] + X_e[2][0])]])


    if(r[i] == 0):
        beacon1 = False
        beacon2 = False
        B.append(0)

    elif (abs(0 - landmark_loc[0][0]) < abs(10 - landmark_loc[0][0])):
        beacon1 = True
        beacon2 = False
        B.append(1)
    else:
        beacon2 = True
        beacon1 = False
        B.append(2)
    
    # Check if B1 is in the field of view
    if(r[i] != 0 and beacon2):
        # Measures with the actual robot state, z=h(X)
        distp_e_2 = np.sqrt((xp2 - xr_e)**2 + (yp2- yr_e)**2)
        psi_p_e_2 = ang_normalized(np.arctan2(yp2 - yr_e, xp2 - xr_e) - theta_r_e)

        z_e = np.array([[distp_e_2],
            [psi_p_e_2]])
    
        z = np.array([[r[i]],
                [psi[i]]])

        # Covariance for the measures 
        R = [[sdv_r**2, 0],
                [0, sdv_psi**2]]
               

        # Gradient of h(X)
        grad_h_X  = np.array([[-((xp2 - xr_e)/(np.sqrt((xp2-xr_e)**2+(yp2-yr_e)**2))), -((yp2-yr_e)/(np.sqrt((xp2-xr_e)**2+(yp2-yr_e)**2))), 0],
                [((yp2-yr_e)/((xp2-xr_e)**2+(yp2-yr_e)**2)), -((xp2-xr_e)/((xp2-xr_e)**2+(yp2-yr_e)**2)), -1]])

        # Kalman Gain
        k = P @ grad_h_X.transpose() @ np.linalg.inv(grad_h_X @ P @ grad_h_X.transpose() + R)

        # Covariance update
        P = (np.eye(3)- k @ grad_h_X) @ P

        # State update
        z_dif = z - z_e
        z_dif[1] = ang_normalized(z_dif[1])
        X_e = X_e + k @ (z_dif)
        X_e[2] = ang_normalized(X_e[2])

    # Check if B2 is in the field of view
    elif(r[i] != 0 and beacon1):
        # Measures with the actual robot state, z=h(X)
        distp_e_1 = np.sqrt((xp1 - xr_e)**2 + (yp1 - yr_e)**2)
        psi_p_e_1 = ang_normalized(np.arctan2(yp1 - yr_e, xp1 - xr_e) - theta_r_e)

        z_e = np.array([[distp_e_1],
                [psi_p_e_1]])
    
        z = np.array([[r[i]],
                [psi[i]]])

        # Covariance for the measures 
        R = [[sdv_r**2, 0],
                [0, sdv_psi**2]]
               

        # Gradient of h(X)
        grad_h_X  = np.array([[-((xp1 - xr_e)/(np.sqrt((xp1-xr_e)**2+(yp1-yr_e)**2))), -((yp1-yr_e)/(np.sqrt((xp1-xr_e)**2+(yp1-yr_e)**2))), 0],
                [((yp1-yr_e)/((xp1-xr_e)**2+(yp1-yr_e)**2)), -((xp1-xr_e)/((xp1-xr_e)**2+(yp1-yr_e)**2)), -1]])

        # Kalman Gain
        k = P @ grad_h_X.transpose() @ np.linalg.inv(grad_h_X @ P @ grad_h_X.transpose() + R)

        # State update
        z_dif = z - z_e
        z_dif[1] = ang_normalized(z_dif[1])
        X_e = X_e + k @ (z_dif)
        X_e[2] = ang_normalized(X_e[2])

        P = P - k @ (grad_h_X @ P @ grad_h_X.T + R) @ k.T
    

    # Save the results in lists 
    X_t.append(X) 
    X_e_t.append(X_e)


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



# Create a function to update the plot in each animation frame
def update(frame):
    plt.clf()  # Clear the previous frame
    plt.subplot(121)  # Subplot on the left
    plt.scatter(x_real, y_real, label='Real Robot Position', color='b', s=5)
    plt.scatter(x_est[:frame], y_est[:frame], label='Robot Position Estimation', color='r', s=5, linestyle='-')
    
    detected_1 = 'yellow' if B[frame] == 1 else 'k'
    detected_2 = 'orange' if B[frame] == 2 else 'k'
 
    plt.scatter(xp1, yp1, label='Beacon 1 Coordinates', color=detected_1, marker='s')
    plt.scatter(xp2, yp2, label='Beacon 2 Coordinates', color=detected_2, marker='s')    
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Actual Robot Position Over Time')
    plt.text(-5, 0, f'Frame: {frame}', fontsize=12, color='black')  # Add frame number as text
    plt.legend()
    plt.grid(True)

    plt.subplot(122)  # Subplot on the right
    plt.scatter(x_real[:frame], y_real[:frame], color='red', label='Real Trajectory', s=5)
    plt.scatter(x_est[:frame], y_est[:frame], color='green', label='Estimated Trajectory', s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

# Create a figure with two subplots
plt.figure(figsize=(12, 6))

# Create the initial plot
ani = FuncAnimation(plt.gcf(), update, frames=len(x_real), repeat=False, interval=50)

# Show the animation
plt.show()

# Create a plot for 'x' vs. 'y'
plt.figure(figsize=(8, 6))
plt.plot(x_real, y_real, label='Real Robot Position', color='b')
plt.plot(x_est, y_est, label='Robot Position Estimation', color='r')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Actual Robot Position Over Time')
plt.legend()
plt.grid(True)
plt.show()


