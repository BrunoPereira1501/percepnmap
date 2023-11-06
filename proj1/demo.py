import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math

sigma_v = 0.5
sigma_w = 0.05

sigma_r = 0.5
sigma_psi = 0.1

landmark1_x = 0
landmark1_y = 0

landmark2_x = 10
landmark2_y = 0


# Read the data from data1.txt
data = np.loadtxt('Datasets-20231106/data1.txt')

# Initialize state vector and covariance matrix
initialPosition = ([[data[0][1]],
                    [data[0][2]],
                    [data[0][3]]])

P = np.eye(3) * 10 ** (-3)

Q = np.array([[sigma_v ** 2, 0],
              [0, sigma_w ** 2]])

R = np.array([[sigma_r ** 2, 0],
              [0, sigma_psi ** 2]])

X = []
X.append(initialPosition)

x_real = []
y_real = []

def normalizeAngle(angle):

  return (angle + math.pi) % (2 * math.pi) - math.pi

for i, row in enumerate(data):

  t, xreal, yreal, _, v, w, r1, psi1, r2, psi2 = row

  x_k = X[i][0][0]
  y_k = X[i][1][0]
  theta_k = X[i][2][0]

  x = x_k + v / w * (math.sin(theta_k + w * 0.1) - math.sin(theta_k))
  y = y_k + v / w * (math.cos(theta_k) - math.cos(theta_k + w * 0.1))
  theta = theta_k + w * 0.1

  X_k = np.array([[x],
                  [y],
                  [theta]])

  fX = np.array([[1, 0, v / w * (math.cos(theta_k + w * 0.1) - math.cos(theta_k))],
                 [0, 1, v / w * (math.sin(theta_k + w * 0.1) - math.sin(theta_k))],
                 [0, 0, 1]])


  fW = np.array([[(math.sin(theta_k + w * 0.1) - math.sin(theta_k)) / w, v * (w * 0.1 * math.cos(theta_k + w * 0.1) + math.sin(theta_k) - math.sin(theta_k + w * 0.1)) / (w ** 2)],
                 [(math.cos(theta_k) - math.cos(theta_k + w * 0.1)) / w, v * (w * 0.1 * math.sin(theta_k + w * 0.1) + math.cos(theta_k + w * 0.1) - math.cos(theta_k)) / (w ** 2)],
                 [0, 0.1]])

  P = fX @ P @ fX.T + fW @ Q @ fW.T

  for j in range(2):

    if j == 0:
      r, psi, landmark_x, landmark_y = r1, psi1, landmark1_x, landmark1_y
    else:
      r, psi, landmark_x, landmark_y = r2, psi2, landmark2_x, landmark2_y

    delta_x = landmark_x - X_k[0][0]
    delta_y = landmark_y - X_k[1][0]
    landmark_distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

    H = np.array([[landmark_distance],
                  [normalizeAngle(math.atan2(delta_y, delta_x) - X_k[2][0])]])

    fH = np.array([[- delta_x / landmark_distance, - delta_y / landmark_distance, 0],
                   [delta_y / (landmark_distance ** 2), - delta_x / (landmark_distance ** 2), -1]])

    K = P @ fH.T @ np.linalg.inv(fH @ P @ fH.T + R)

    z = np.array([[r],
                  [psi]])

  
    z_dif = z - H
    z_dif[1] = normalizeAngle(z_dif[1])
    X_k = X_k + K @ (z_dif)
    X_k[2] = normalizeAngle(X_k[2])
    P = P - K @ (fH @ P @ fH.T + R) @ K.T

  x_real.append(xreal)
  y_real.append(yreal)
  X.append(X_k)

x_position = []
y_position = []



for state in X:
  x_position.append(state[0])
  y_position.append(state[1])

print(x_position)

plt.figure(figsize=(8, 6))
plt.plot(x_position, y_position, label = 'Estimated Trajectory', color = 'blue')
plt.plot(x_real, y_real, label = "Real Trajectory", color = 'black')
plt.scatter([landmark1_x, landmark2_x], [landmark1_y, landmark2_y], marker='o', color='red', label='Landmarks')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Estimated Robot Trajectory')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()