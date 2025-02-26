# -*- coding: utf-8 -*-
"""ECGR 4105.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18ONd_VNJ3D1gK4Rkfa93H-29N0LMvn3C
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('D3.csv')
X = data[['X1', 'X2', 'X3']].values
y = data['Y'].values

# Set the learning rate and number of iterations
alpha = 0.01  # You can experiment with different values like 0.1
iterations = 1000

# Define the linear regression function (modified to return theta history)
def linear_regression(X, y, alpha, iterations):
    m = len(y)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # Add bias term
    theta = np.zeros(X.shape[1])  # Initialize theta to zero
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))  # Store theta history

    for i in range(iterations):
        h = np.dot(X, theta)  # Hypothesis
        loss = h - y  # Calculate the error
        gradient = np.dot(X.T, loss) / m  # Compute the gradient
        theta -= alpha * gradient  # Update theta
        cost = np.sum(loss**2) / (2 * m)  # Calculate the cost (for monitoring)
        cost_history[i] = cost
        theta_history[i] = theta.copy()  # Store theta at each iteration

    return theta, cost_history, theta_history

# Train the model for each explanatory variable in isolation and plot
theta_history_all = []
loss_history_all = []  # Store loss history
X_train_list = []  # Store the training data for plotting

for i in range(X.shape[1]):
    X_train = X[:, i].reshape(-1, 1)  # Select one explanatory variable
    X_train_list.append(X_train)  # Store the training data
    theta, cost, theta_history = linear_regression(X_train, y, alpha, iterations)
    theta_history_all.append(theta_history)

    # Calculate loss for the last iteration
    X_train_with_bias = np.concatenate((np.ones((len(X_train), 1)), X_train), axis=1)
    h = np.dot(X_train_with_bias, theta_history[-1]) # Use the final theta values
    loss = h - y
    loss_history_all.append(loss)

    # Plot the loss (separate plot)
    plt.figure()  # Create a new figure for each plot
    plt.scatter(X_train, loss)  # Scatter plot of X vs. loss
    plt.xlabel(f"X{i+1}")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. X{i+1} (Final Iteration)")
    plt.show()

    # Plot the final regression model (separate plot)
    y_pred = np.dot(X_train_with_bias, theta)  # Use the final theta
    plt.figure()  # Create a new figure for each plot
    plt.scatter(X_train, y, label="Data")
    plt.plot(X_train, y_pred, color='red', label="Regression Line")
    plt.xlabel(f"X{i+1}")
    plt.ylabel("Y")
    plt.title(f"Regression Line (X{i+1})")
    plt.legend()
    plt.show()

# Report the linear models
print("Linear Models:")
for i in range(X.shape[1]):
    print(f"Y = {theta_history_all[i][-1][0]:.4f} + {theta_history_all[i][-1][1]:.4f} * X{i+1}")  # Access last theta values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('D3.csv')
X = data[['X1', 'X2', 'X3']].values
y = data['Y'].values

# Set the learning rate and number of iterations
alpha = 0.05  # Experiment with different alpha values
iterations = 10000

# Define the linear regression function (modified to return theta history)
def linear_regression(X, y, alpha, iterations):
    m = len(y)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # Add bias term
    theta = np.zeros(X.shape[1])  # Initialize theta to zero
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))  # Store theta history

    for i in range(iterations):
        h = np.dot(X, theta)  # Hypothesis
        loss = h - y  # Calculate the error
        gradient = np.dot(X.T, loss) / m  # Compute the gradient
        theta -= alpha * gradient  # Update theta
        cost = np.sum(loss**2) / (2 * m)  # Calculate the cost (for monitoring)
        cost_history[i] = cost
        theta_history[i] = theta.copy()  # Store theta at each iteration

    return theta, cost_history, theta_history


# Train the model using all 3 variables at once
theta, cost_history, theta_history_all = linear_regression(X, y, alpha, iterations)  # Train with all X

# Calculate the loss for the final iteration
X_with_bias = np.concatenate((np.ones((len(X), 1)), X), axis=1)
h = np.dot(X_with_bias, theta)  # Predictions using the final theta
loss = h - y

# Plot the loss vs. data index
plt.figure()
plt.scatter(range(len(loss)), loss)  # Scatter plot of loss vs. data index
plt.xlabel("Data Point Index")
plt.ylabel("Loss")
plt.title("Loss vs. Data Point Index (Final Iteration)")
plt.show()


# Plot the regression plane (for 3D data)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Create a meshgrid of X values for the plane
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# Calculate predicted Y values for the plane (X3 is 0 for visualization)
X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten(), np.zeros_like(x1_mesh.flatten())))
X_mesh_with_bias = np.concatenate((np.ones((len(X_mesh), 1)), X_mesh), axis=1)
y_pred_mesh = np.dot(X_mesh_with_bias, theta)
y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

# Plot the plane and the data points
ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.5)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, label="Data")  # Color points by y value
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title("Regression Plane (All Variables)")
ax.legend()
plt.show()

# Report the linear model
print("Linear Model (All Variables):")
print(f"Y = {theta[0]:.4f} + {theta[1]:.4f} * X1 + {theta[2]:.4f} * X2 + {theta[3]:.4f} * X3")

