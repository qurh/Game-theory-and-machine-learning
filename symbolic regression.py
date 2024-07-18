# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:02:08 2024

@author: User
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Define the true ODE system (predator-prey model)
def true_system(X, t, a, b):
    x, y = X
    dxdt = a * x - b * x * y
    dydt = -a * y + b * x * y
    return [dxdt, dydt]

# Generate synthetic data
def generate_data(a, b, x0, y0, t):
    X0 = [x0, y0]
    solution = odeint(true_system, X0, t, args=(a, b))
    return solution

# Parameters
a, b = 1.0, 0.1
x0, y0 = 5.0, 3.0
t = np.linspace(0, 20, 200)

# Generate data
data = generate_data(a, b, x0, y0, t)

# Calculate derivatives
dXdt = np.gradient(data, t, axis=0)

# Prepare data for symbolic regression
X = data
y_dx = dXdt[:, 0]
y_dy = dXdt[:, 1]

# Define custom function for multiplication
def protected_multiplication(x1, x2):
    return x1 * x2

mult = make_function(function=protected_multiplication,
                     name='mul',
                     arity=2)

# Create and train symbolic regressors
function_set = ['add', 'sub', mult]
sr_dx = SymbolicRegressor(population_size=5000,
                          generations=10,
                          function_set=function_set,
                          metric='mse',
                          p_crossover=0.7,
                          p_subtree_mutation=0.1,
                          p_hoist_mutation=0.05,
                          p_point_mutation=0.1,
                          max_samples=0.9,
                          verbose=1,
                          parsimony_coefficient=0.01,
                          random_state=0)

sr_dy = SymbolicRegressor(population_size=5000,
                          generations=10,
                          function_set=function_set,
                          metric='mse',
                          p_crossover=0.7,
                          p_subtree_mutation=0.1,
                          p_hoist_mutation=0.05,
                          p_point_mutation=0.1,
                          max_samples=0.9,
                          verbose=1,
                          parsimony_coefficient=0.01,
                          random_state=0)

# Fit the models
sr_dx.fit(X, y_dx)
sr_dy.fit(X, y_dy)

# Print the learned equations
print("Learned dx/dt equation:", sr_dx._program)
print("Learned dy/dt equation:", sr_dy._program)

# Function to compute predictions using the learned models
def learned_system(X, t):
    x, y = X
    dxdt = sr_dx.predict([[x, y]])
    dydt = sr_dy.predict([[x, y]])
    return [dxdt[0], dydt[0]]

# Generate predictions using the learned model
X_pred = odeint(learned_system, [x0, y0], t)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, data[:, 0], label='True x')
plt.plot(t, data[:, 1], label='True y')
plt.plot(t, X_pred[:, 0], '--', label='Predicted x')
plt.plot(t, X_pred[:, 1], '--', label='Predicted y')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('ODE System: True vs Predicted (Symbolic Regression)')
plt.show()

# Visualize the vector fields
x_range = np.linspace(0, 6, 20)
y_range = np.linspace(0, 4, 20)
X, Y = np.meshgrid(x_range, y_range)

# True vector field
U_true, V_true = true_system([X, Y], 0, a, b)

# Learned vector field
U_learned = sr_dx.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
V_learned = sr_dy.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(Y.shape)

# Plot vector fields
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# True vector field
strm1 = ax1.streamplot(X, Y, U_true, V_true, density=1, linewidth=1, arrowsize=1, arrowstyle='->')
ax1.set_xlabel('x (Prey Population)')
ax1.set_ylabel('y (Predator Population)')
ax1.set_title('True Vector Field')
fig.colorbar(strm1.lines, ax=ax1, label='Magnitude of Change')

# Learned vector field
strm2 = ax2.streamplot(X, Y, U_learned, V_learned, density=1, linewidth=1, arrowsize=1, arrowstyle='->')
ax2.set_xlabel('x (Prey Population)')
ax2.set_ylabel('y (Predator Population)')
ax2.set_title('Learned Vector Field (Symbolic Regression)')
fig.colorbar(strm2.lines, ax=ax2, label='Magnitude of Change')

plt.tight_layout()
plt.show()

# Calculate and plot error
error_magnitude = np.sqrt((U_learned - U_true)**2 + (V_learned - V_true)**2)

plt.figure(figsize=(12, 8))
contour = plt.contourf(X, Y, error_magnitude, levels=20, cmap='viridis')
plt.colorbar(contour, label='Error Magnitude')
plt.xlabel('x (Prey Population)')
plt.ylabel('y (Predator Population)')
plt.title('Error Between True and Learned ODE Systems (Symbolic Regression)')
plt.show()