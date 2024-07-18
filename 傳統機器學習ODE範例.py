
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Main simulation and fitting process
a, b = 1.0, 0.1
x0, y0 = 5.0, 3.0
t = np.linspace(0, 20, 200)

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

# Define basis functions
def basis_functions(X):
    x, y = X[:, 0], X[:, 1]
    return np.column_stack([x, y, x*y])

# Custom fitting function
def fit_model_custom(X, dXdt, objective_function, initial_params = np.array([a, 0, -b, 0, -a, b])):
    basis = basis_functions(X)
    
    if initial_params is None:
        initial_params = np.zeros(basis.shape[1] * 2)  # For both x and y equations
    
    def loss(params):
        params_x = params[:basis.shape[1]]
        params_y = params[basis.shape[1]:]
        dxdt_pred = basis @ params_x
        dydt_pred = basis @ params_y
        return objective_function(dXdt, np.column_stack([dxdt_pred, dydt_pred]))
    
    result = minimize(loss, initial_params, method='BFGS')
    
    model_x_params = result.x[:basis.shape[1]]
    model_y_params = result.x[basis.shape[1]:]
    
    return model_x_params, model_y_params

# Predict using the fitted models
def predict(X, model_x_params, model_y_params):
    basis = basis_functions(X)
    dxdt = basis @ model_x_params
    dydt = basis @ model_y_params
    return np.column_stack([dxdt, dydt])

# Custom objective function
def custom_objective(dXdt_true, dXdt_pred):
    # This is a simple MSE. Modify this function as needed for your research.
    mse = np.mean((dXdt_true - dXdt_pred)**2)
    # Add custom terms here, e.g., regularization, constraints, etc.
    return mse

# Generate synthetic data
data = generate_data(a, b, x0, y0, t)

# Calculate derivatives
dXdt = np.gradient(data, t, axis=0)

# Fit the models using custom objective function
model_x_params, model_y_params = fit_model_custom(data, dXdt, custom_objective)

# Print the estimated parameters and reconstructed ODE system
print("Estimated parameters for dx/dt:")
print(f"a1 = {model_x_params[0]:.4f}, a2 = {model_x_params[1]:.4f}, a3 = {model_x_params[2]:.4f}")
print("\nEstimated parameters for dy/dt:")
print(f"b1 = {model_y_params[0]:.4f}, b2 = {model_y_params[1]:.4f}, b3 = {model_y_params[2]:.4f}")

print("\nReconstructed ODE system:")
print(f"dx/dt = {model_x_params[0]:.4f}x + {model_x_params[1]:.4f}y + {model_x_params[2]:.4f}xy")
print(f"dy/dt = {model_y_params[0]:.4f}x + {model_y_params[1]:.4f}y + {model_y_params[2]:.4f}xy")

print("\nTrue ODE system:")
print(f"dx/dt = {a}x - {b}xy")
print(f"dy/dt = -{a}y + {b}xy")

# Generate predictions using the fitted model
def learned_system(X, t):
    return predict(X.reshape(1, -1), model_x_params, model_y_params).flatten()

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
plt.title('ODE System: True vs Predicted')
plt.show()

# Plot the phase portrait
plt.figure(figsize=(10, 8))
plt.plot(data[:, 0], data[:, 1], label='True Trajectory')
plt.plot(X_pred[:, 0], X_pred[:, 1], '--', label='Predicted Trajectory')

# Add vector field
x_range = np.linspace(0, 6, 20)
y_range = np.linspace(0, 4, 20)
X, Y = np.meshgrid(x_range, y_range)
UV = predict(np.column_stack([X.ravel(), Y.ravel()]), model_x_params, model_y_params).reshape(X.shape[0], X.shape[1], 2)

plt.streamplot(X, Y, UV[:,:,0], UV[:,:,1], density=1, linewidth=1, arrowsize=1, arrowstyle='->')
plt.xlabel('x (Prey Population)')
plt.ylabel('y (Predator Population)')
plt.title('Phase Portrait: True vs Predicted Trajectory')
plt.legend()
plt.show()