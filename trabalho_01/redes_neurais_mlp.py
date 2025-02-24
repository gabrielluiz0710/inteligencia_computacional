import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from ucimlrepo import fetch_ucirepo

# Fetch dataset
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

# Normalize data using given formula
Lmin, Lmax = 0, 1
X_min, X_max = X.min(), X.max()
X = ((Lmax - Lmin) * (X - X_min) / (X_max - X_min)) + Lmin

y_min, y_max = y.min(), y.max()
y = ((Lmax - Lmin) * (y - y_min) / (y_max - y_min)) + Lmin

# Split dataset (70% Training, 23% Validation, 7% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.233, random_state=42)

# Define architectures
architectures = [[13, 13, 1], [13, 13, 13, 1], [13, 13, 13, 13, 1]]
results = {}

for arch in architectures:
    # Create MLP model
    mlp = MLPRegressor(hidden_layer_sizes=arch[1:-1], activation='logistic', solver='adam',
                        learning_rate_init=0.1, max_iter=500, random_state=42)
    
    train_errors, val_errors = [], []
    
    # Train model and capture MSE at each epoch
    for epoch in range(500):
        mlp.partial_fit(X_train, y_train.values.ravel())
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_errors.append(mean_squared_error(y_train, train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))
    
    # Final Predictions
    y_pred = mlp.predict(X_test)
    
    # Compute error metrics
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    cv = np.std(y_pred) / np.mean(y_pred)
    
    results[str(arch)] = {
        'mse': mse,
        'mape': mape,
        'cv': cv,
        'train_errors': train_errors,
        'val_errors': val_errors
    }
    
    # Plot training curve
    plt.figure()
    plt.plot(train_errors, label='Training MSE')
    plt.plot(val_errors, label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title(f'Training Curve for Architecture {arch}')
    plt.legend()
    plt.show()

# Display results
for arch, metrics in results.items():
    print(f'Architecture {arch}:')
    print(f'  Mean Squared Error: {metrics["mse"]:.6f}')
    print(f'  Mean Absolute Percentage Error: {metrics["mape"]:.6f}')
    print(f'  Coefficient of Variation: {metrics["cv"]:.6f}\n')
