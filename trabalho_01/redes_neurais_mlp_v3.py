import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo 

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets

# Normalização [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Divisão dos dados (70% treino, 23% validação, 7% teste)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.233, random_state=42)

# Função para criar e treinar MLP
def train_mlp(layers, learning_rate=0.1, epochs=5000):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(13,)))  # Camada de entrada
    
    # Camadas ocultas
    for _ in range(len(layers) - 2):
        model.add(tf.keras.layers.Dense(13, activation='sigmoid'))
    
    # Camada de saída
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error', metrics=['mae'])
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)
    
    return model, history

# Treinar diferentes arquiteturas
architectures = {
    "MLP 1 Camada Oculta": [13, 13, 1],
    "MLP 2 Camadas Ocultas": [13, 13, 13, 1],
    "MLP 3 Camadas Ocultas": [13, 13, 13, 13, 1]
}

results = {}
for name, layers in architectures.items():
    model, history = train_mlp(layers)
    results[name] = {
        "EQM": np.mean(history.history['loss']),
        "EPMA": np.mean(history.history['mae']),
        "Modelo": model
    }

# Calcular Coeficiente de Variação
def coef_variacao(model, X_test):
    predicoes = np.array([model.predict(X_test) for _ in range(20)])
    media = np.mean(predicoes, axis=0)
    desvio_padrao = np.std(predicoes, axis=0)
    return np.mean(desvio_padrao / media)

for name, data in results.items():
    results[name]["CV"] = coef_variacao(data["Modelo"], X_test)

# Exibir resultados
for name, metrics in results.items():
    print(f"{name} -> EQM: {metrics['EQM']:.6f}, EPMA: {metrics['EPMA']:.6f}, CV: {metrics['CV']:.6f}")
