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

# Initialize an empty list to store the data
data_list = []

# Open the data file for reading
with open('Datasets-20231026\data4.txt', 'r') as file:
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
Q = np.array([[0.5**2, 0],
     [0, 0.05**2]])

# Std for the beacons 
sdv_r = 0.5
sdv_psi = 0.1

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
X_e = [[0], [0], [0]]
X_e_t.append(X_e)


# Create list for landmarks
landmarks = []
nr_landmarks = 0
state_dim = 3 + 2 * nr_landmarks
P = np.eye(state_dim)*1e-3

# Sampling period
dt = 0.1

for i in range(int(N)):
    
    # Real Trajectory
    X = [[x[i]], [y[i]], [theta[i]]]
     
    # Predict X(k+1) = f(X(k),U)
    xr_e = X_e_t[i][0][0]
    yr_e = X_e_t[i][1][0]
    theta_r_e = X_e_t[i][2][0]

    mat = np.eye(3)
    zero_columns = np.zeros((mat.shape[0], nr_landmarks*2))
    FxT = np.hstack((mat, zero_columns))


    Mot = np.array([[v[i]/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e))],
            [v[i]/w[i] * (np.cos(theta_r_e) - np.cos(theta_r_e + w[i]*dt))],
            [w[i] * dt]])  
    
    X_e = X_e + FxT.T @ Mot
    

    # Gradient of f(X)
    grad_f_X = np.array([[1, 0, v[i]/w[i] * (np.cos(theta_r_e + w[i]*dt) - np.cos(theta_r_e))],
            [0, 1, v[i]/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e))],
            [0, 0, 1]])
    
    if(state_dim == 3):
        Gt = grad_f_X
    else:
        expanded = np.eye((state_dim))
        expanded[0:3, 0:3] = grad_f_X
        Gt = expanded
    
    # Gradient of f(U)
    grad_f_U = np.array([[1/w[i] * (np.sin(theta_r_e + w[i]*dt) - np.sin(theta_r_e)), v[i]/(w[i]**2) * (dt*w[i]*np.cos(theta_r_e + w[i]*dt) - np.sin(theta_r_e + w[i]*dt)+ np.sin(theta_r_e))],
            [1/w[i] * (-np.cos(theta_r_e + w[i]*dt) + np.cos(theta_r_e)), v[i]/(w[i]**2) * (dt*w[i]*np.sin(theta_r_e + w[i]*dt) + np.cos(theta_r_e + w[i]*dt) - np.cos(theta_r_e))],
            [0, dt]])
    
    if(nr_landmarks == 0):
        Ut = grad_f_U
    else:
        zero_rows = np.zeros((nr_landmarks*2, 2))
        Ut = np.vstack((grad_f_U, zero_rows))

    # Covariance propagation
    P = Gt @ P @ Gt.T + Ut @ Q @ Ut.T

    if(r[i] != 0.0):    
       
        landmark_loc = np.array([[X_e[0][0] + r[i] * np.cos(psi[i] + X_e[2][0])],
                                 [X_e[1][0] + r[i] * np.sin(psi[i] + X_e[2][0])]])
        

        if(nr_landmarks == 0):

            landmarks.append(landmark_loc)
            nr_landmarks += 1
            state_dim = 3 + 2 * nr_landmarks

            X_e = np.vstack((X_e, landmark_loc))
            expanded = np.eye(state_dim) #* 10
            expanded[0:state_dim-2, :state_dim-2] = P
            P = expanded

            distp_e = np.sqrt((landmark_loc[0][0] - X_e[0][0])**2 + (landmark_loc[1][0] - X_e[1][0])**2)
            psi_p_e = ang_normalized(np.arctan2(landmark_loc[1][0] - X_e[1][0], landmark_loc[0][0] - X_e[0][0]) - X_e[2][0])

            z_e = np.array([[distp_e],
                    [psi_p_e]])

            z = np.array([[r[i]],
                    [psi[i]]])

            # Covariance for the measures 
            R = [[sdv_r**2, 0],
                    [0, sdv_psi**2]]

            # Gradient of h(X)
            grad_h_X  = np.array([[-((landmark_loc[0][0] - X_e[0][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), -((landmark_loc[1][0]-X_e[1][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), 0, ((landmark_loc[0][0] - X_e[0][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), ((landmark_loc[1][0]-X_e[1][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)))],
                    [((landmark_loc[1][0]-X_e[1][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), -((landmark_loc[0][0]-X_e[0][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), -1, -((landmark_loc[1][0]-X_e[1][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), ((landmark_loc[0][0]-X_e[0][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))]])
            
            FxJ = grad_h_X
       
            # Kalman Gain
            k = P @ FxJ.T @ np.linalg.inv(FxJ @ P @ FxJ.T + R)
        
            # Covariance update
            P = (np.eye(state_dim) - k @ FxJ) @ P

            # State update
            z_dif = z - z_e
            z_dif[1] = ang_normalized(z_dif[1])
            X_e = X_e + k @ (z_dif)
            X_e[2] = ang_normalized(X_e[2])

        else:
            # If there are known beacons, associate the measurement with the closest one
            min_distance = float("inf")
            closest_landmark_idx = -1

            for j, landmark in enumerate(landmarks):
                distance = np.sqrt((landmark_loc[0][0] - landmark[0]) ** 2 + (landmark_loc[1][0] - landmark[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_landmark_idx = j

            # Check if the measurement corresponds to a known landmark or a new one
            if min_distance < 5:
                # If associated with a known beacon, update that landmark
                closest_landmark = landmarks[closest_landmark_idx]
                # Update the associated landmark logic here

                distp_e = np.sqrt((landmark_loc[0][0] - X_e[0][0])**2 + (landmark_loc[1][0] - X_e[1][0])**2)
                psi_p_e = ang_normalized(np.arctan2(landmark_loc[1][0] - X_e[1][0], landmark_loc[0][0] - X_e[0][0]) - X_e[2][0])

                z_e = np.array([[distp_e],
                        [psi_p_e]])

                z = np.array([[r[i]],
                        [psi[i]]])

                # Covariance for the measures 
                R = [[sdv_r**2, 0],
                        [0, sdv_psi**2]]

                # Gradient of h(X)
                grad_h_X  = np.array([[-((landmark_loc[0][0] - X_e[0][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), -((landmark_loc[1][0]-X_e[1][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), 0, ((landmark_loc[0][0] - X_e[0][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), ((landmark_loc[1][0]-X_e[1][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)))],
                        [((landmark_loc[1][0]-X_e[1][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), -((landmark_loc[0][0]-X_e[0][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), -1, -((landmark_loc[1][0]-X_e[1][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), ((landmark_loc[0][0]-X_e[0][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))]])
                
                if(nr_landmarks == 1):
                    FxJ = grad_h_X
                else:
                    iden = np.eye(5)
                    zero_mat_before = np.zeros((5, 2*j-2))
                    zero_mat_after = np.zeros((5, 2*nr_landmarks-2*j))
                    FxJ = np.insert(iden, [3], zero_mat_before, axis = 1)
                    FxJ = np.insert(FxJ, [5 + (2*j-2)], zero_mat_after, axis = 1)
                    FxJ = grad_h_X @ FxJ

                # Kalman Gain
                k = P @ FxJ.T @ np.linalg.inv(FxJ @ P @ FxJ.T + R)

                # Covariance update
                P = (np.eye(state_dim) - k @ FxJ) @ P

                # State update
                z_dif = z - z_e
                z_dif[1] = ang_normalized(z_dif[1])
                X_e = X_e + k @ (z_dif)
                X_e[2] = ang_normalized(X_e[2])

            else:
                j+=1
                # If not associated with any known beacon, add as a new landmark
                landmarks.append(landmark_loc)
                nr_landmarks += 1
                state_dim = 3 + 2 * nr_landmarks

                # Expand state and covariance matrices
                X_e = np.vstack((X_e, landmark_loc))
                expanded = np.eye(state_dim) * 10  # Adjust covariance initialization as needed
                expanded[0:state_dim - 2, 0:state_dim - 2] = P
                P = expanded

                distp_e = np.sqrt((landmark_loc[0][0] - X_e[0][0])**2 + (landmark_loc[1][0] - X_e[1][0])**2)
                psi_p_e = ang_normalized(np.arctan2(landmark_loc[1][0] - X_e[1][0], landmark_loc[0][0] - X_e[0][0]) - X_e[2][0])

                z_e = np.array([[distp_e],
                        [psi_p_e]])

                z = np.array([[r[i]],
                        [psi[i]]])

                # Covariance for the measures 
                R = [[sdv_r**2, 0],
                        [0, sdv_psi**2]]

                # Gradient of h(X)
                grad_h_X  = np.array([[-((landmark_loc[0][0] - X_e[0][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), -((landmark_loc[1][0]-X_e[1][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), 0, ((landmark_loc[0][0] - X_e[0][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))), ((landmark_loc[1][0]-X_e[1][0])/(np.sqrt((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)))],
                        [((landmark_loc[1][0]-X_e[1][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), -((landmark_loc[0][0]-X_e[0][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), -1, -((landmark_loc[1][0]-X_e[1][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2)), ((landmark_loc[0][0]-X_e[0][0])/((landmark_loc[0][0]-X_e[0][0])**2+(landmark_loc[1][0]-X_e[1][0])**2))]])
                
                if(nr_landmarks == 1):
                    FxJ = grad_h_X
                else:
                    iden = np.eye(5)
                    zero_mat_before = np.zeros((5, 2*j-2))
                    zero_mat_after = np.zeros((5, 2*nr_landmarks-2*j))
                    FxJ = np.insert(iden, [3], zero_mat_before, axis = 1)
                    FxJ = np.insert(FxJ, [5 + (2*j-2)], zero_mat_after, axis = 1)
                    FxJ = grad_h_X @ FxJ

                # Kalman Gain
                k = P @ FxJ.T @ np.linalg.inv(FxJ @ P @ FxJ.T + R)

                # Covariance update
                P = (np.eye(state_dim) - k @ FxJ) @ P

                # State update
                z_dif = z - z_e
                z_dif[1] = ang_normalized(z_dif[1])
                X_e = X_e + k @ (z_dif)
                X_e[2] = ang_normalized(X_e[2])
                

    # Save the results in lists 
    X_t.append(X) 
    X_e_t.append(X_e)
            
print(landmarks)

# Initialize empty lists for x and y values
x_real = []
y_real = []
x_est = []
y_est = []
x_beacons = []
y_beacons = []

# Extract x and y values from each array in the list
for array in X_t:
    x_real.append(array[0])  # Extract x (first element)
    y_real.append(array[1])  # Extract y (second element)

# Extract x and y values from each array in the list
for array in X_e_t:
    x_est.append(array[0])  # Extract x (first element)
    y_est.append(array[1])  # Extract y (second element)

# Extract x and y values from each array in the list
for array in landmarks:
    x_beacons.append(array[0])  # Extract x (first element)
    y_beacons.append(array[1])  # Extract y (second element)



# Create a function to update the plot in each animation frame
def update(frame):
    plt.clf()  # Clear the previous frame
    plt.subplot(121)  # Subplot on the left
    plt.scatter(x_real, y_real, label='Real Robot Position', color='b', s=5)
    plt.scatter(x_est[:frame], y_est[:frame], label='Robot Position Estimation', color='r', s=5, linestyle='-')
    plt.scatter(x_beacons, y_beacons, label='Possible Beacon Coordinates', color='orange', marker='s')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Actual Robot Position Over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)  # Subplot on the right
    plt.scatter(x_real[:frame], y_real[:frame], color='red', label='Real Trajectory', s=5)
    plt.scatter(x_est[:frame], y_est[:frame], color='green', label='Estimated Trajectory', s=5)
    plt.text(-5, 0, f'Frame: {frame}', fontsize=12, color='black')  # Add frame number as text
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

# Create a figure with two subplots
plt.figure(figsize=(12, 6))

# Create the initial plot
ani = FuncAnimation(plt.gcf(), update, frames=len(x_real), repeat=False, interval=1)

# Show the animation
plt.show()

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

