# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:58:07 2024

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# Define the true ODE system
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

# Define the neural network model
class ODENet(nn.Module):
    def __init__(self):
        super(ODENet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Flexible loss function
def flexible_loss(model, X, dXdt_true, loss_config):
    dXdt_pred = model(X)
    
    # MSE Loss
    mse_loss = torch.mean((dXdt_pred - dXdt_true) ** 2)
    
    # L1 Loss
    l1_loss = torch.mean(torch.abs(dXdt_pred - dXdt_true))
    
    # Conservation law
    conservation_loss = torch.mean((dXdt_pred[:, 0] + dXdt_pred[:, 1]) ** 2)
    
    # Smoothness loss
    smoothness_loss = torch.mean(torch.abs(torch.diff(dXdt_pred, dim=0)))
    
    # Combine losses
    total_loss = (
        loss_config['mse_weight'] * mse_loss +
        loss_config['l1_weight'] * l1_loss +
        loss_config['conservation_weight'] * conservation_loss +
        loss_config['smoothness_weight'] * smoothness_loss
    )
    
    return total_loss, {
        'mse_loss': mse_loss.item(),
        'l1_loss': l1_loss.item(),
        'conservation_loss': conservation_loss.item(),
        'smoothness_loss': smoothness_loss.item(),
        'total_loss': total_loss.item()
    }

# Training function
def train_model(model, X_train, y_train, loss_config, num_epochs=1000, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss, loss_components = flexible_loss(model, X_train, y_train[:, 2:], loss_config)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss_components)
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss_history

# Function to compute the vector field
def compute_vector_field(model, x_range, y_range):
    X, Y = np.meshgrid(x_range, y_range)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    with torch.no_grad():
        UV = model(torch.tensor(XY, dtype=torch.float32)).numpy()
    U = UV[:, 0].reshape(X.shape)
    V = UV[:, 1].reshape(Y.shape)
    return X, Y, U, V

# Main simulation and training process
a, b = 1.0, 0.1
x0, y0 = 5.0, 3.0
t = np.linspace(0, 20, 200)

# Generate synthetic data
data = generate_data(a, b, x0, y0, t)

# Calculate derivatives
dXdt = np.gradient(data, t, axis=0)

# Prepare training data
X_train = torch.tensor(data, dtype=torch.float32)
y_train = torch.tensor(np.concatenate([data, dXdt], axis=1), dtype=torch.float32)

# Create the model
model = ODENet()

# Define loss configuration
loss_config = {
    'mse_weight': 1.0,
    'l1_weight': 0.1,
    'conservation_weight': 0.01,
    'smoothness_weight': 0.01
}

# Train the model
loss_history = train_model(model, X_train, y_train, loss_config)

# Generate predictions using the trained model
def nn_system(X, t):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

X_pred = odeint(nn_system, [x0, y0], t)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, data[:, 0], label='True x')
plt.plot(t, data[:, 1], label='True y')
plt.plot(t, X_pred[:, 0], '--', label='Predicted x')
plt.plot(t, X_pred[:, 1], '--', label='Predicted y')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('ODE System: True vs Predicted')
plt.show()

# Plot loss components
loss_df = pd.DataFrame(loss_history)
plt.figure(figsize=(12, 6))
for column in loss_df.columns:
    plt.plot(loss_df[column], label=column)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Components During Training')
plt.yscale('log')
plt.show()

# Visualize the vector fields
x_range = np.linspace(0, 6, 20)
y_range = np.linspace(0, 4, 20)

# True vector field
X, Y = np.meshgrid(x_range, y_range)
U_true, V_true = true_system([X, Y], 0, a, b)

# Learned vector field
X, Y, U_learned, V_learned = compute_vector_field(model, x_range, y_range)

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
ax2.set_title('Learned Vector Field')
fig.colorbar(strm2.lines, ax=ax2, label='Magnitude of Change')

plt.tight_layout()
plt.show()

# Plot the phase portrait
plt.figure(figsize=(12, 8))
plt.plot(data[:, 0], data[:, 1], label='True Trajectory')
plt.plot(X_pred[:, 0], X_pred[:, 1], '--', label='Predicted Trajectory')
plt.streamplot(X, Y, U_learned, V_learned, density=1, linewidth=1, arrowsize=1, arrowstyle='->')
plt.xlabel('x (Prey Population)')
plt.ylabel('y (Predator Population)')
plt.title('Phase Portrait: True vs Predicted Trajectory')
plt.legend()
plt.show()

# Extended analysis

def learned_dynamics(x, y):
    with torch.no_grad():
        input_tensor = torch.tensor([x, y], dtype=torch.float32)
        output = model(input_tensor)
    return output.numpy()

# Create a grid of points
x_range = np.linspace(0, 6, 20)
y_range = np.linspace(0, 4, 20)
X, Y = np.meshgrid(x_range, y_range)

# Sample the learned dynamics
dXdt = np.zeros_like(X)
dYdt = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        dXdt[i,j], dYdt[i,j] = learned_dynamics(X[i,j], Y[i,j])



# Compare with true ODE system
def true_dynamics(X, Y, a=1.0, b=0.1):
    dXdt = a * X - b * X * Y
    dYdt = -a * Y + b * X * Y
    return dXdt, dYdt

dXdt_true, dYdt_true = true_dynamics(X, Y)


# Analyze specific points
sample_points = [(1, 1), (3, 10), (2, 10)]

print("Comparing true and learned dynamics at specific points:")
print("(x, y) | True (dx/dt, dy/dt) | Learned (dx/dt, dy/dt)")
print("-" * 60)

for x, y in sample_points:
    true_deriv = true_dynamics(x, y)
    learned_deriv = learned_dynamics(x, y)
    print(f"({x}, {y}) | ({true_deriv[0]:.3f}, {true_deriv[1]:.3f}) | ({learned_deriv[0]:.3f}, {learned_deriv[1]:.3f})")

# Calculate and plot error
error_magnitude = np.sqrt((dXdt - dXdt_true)**2 + (dYdt - dYdt_true)**2)

plt.figure(figsize=(12, 8))
contour = plt.contourf(X, Y, error_magnitude, levels=20, cmap='viridis')
plt.colorbar(contour, label='Error Magnitude')
plt.xlabel('x (Prey Population)')
plt.ylabel('y (Predator Population)')
plt.title('Error Between True and Learned ODE Systems')
plt.show()