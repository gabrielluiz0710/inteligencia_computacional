import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt

# Função de normalização conforme especificado no artigo
def custom_normalization(X, L_min=0, L_max=1):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Evitar divisão por zero
    denom = X_max - X_min
    denom[denom == 0] = 1  # Se X_max == X_min, evita divisão por zero
    
    return ((L_max - L_min) * (X - X_min) / denom) + L_min

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Data (as pandas dataframes)
X = wine.data.features.values
y = wine.data.targets.values

# Aplicar a normalização personalizada
X = custom_normalization(X)

# Divisão dos dados (70% treino, 23% validação, 7% teste)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.233, random_state=42)

# Função para criar e treinar MLP com SGD e salvar EQM por época
def train_mlp(layers, learning_rate=0.1, epochs=5000):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(13,)))  # Camada de entrada
    
    # Camadas ocultas
    for _ in range(len(layers) - 2):
        model.add(tf.keras.layers.Dense(13, activation='sigmoid'))
    
    # Camada de saída
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
                  loss='mean_squared_error', metrics=['mae'])
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)
    
    return model, history.history['loss'], history.history['val_loss']

# Treinar diferentes arquiteturas
architectures = {
    "MLP 1 Camada Oculta": [13, 13, 1],
    "MLP 2 Camadas Ocultas": [13, 13, 13, 1],
    "MLP 3 Camadas Ocultas": [13, 13, 13, 13, 1]
}

results = {}
for name, layers in architectures.items():
    model, train_eqm, val_eqm = train_mlp(layers)
    
    # Fazer previsões
    y_pred = model.predict(X_test).flatten()
    
    # Calcular métricas
    eqm = np.mean((y_test - y_pred) ** 2)
    nonzero_mask = y_test != 0
    epma = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask]))
        
    results[name] = {
        "EQM": eqm,
        "EPMA": epma,
        "Modelo": model,
        "Train_EQM": train_eqm,
        "Val_EQM": val_eqm
    }

# Calcular Coeficiente de Variação
def coef_variacao(model, X_test):
    predicoes = np.zeros((20, len(X_test)))
    for i in range(20):
        predicoes[i] = model.predict(X_test).flatten()
    media = np.mean(predicoes, axis=0)
    desvio_padrao = np.std(predicoes, axis=0)
    return np.mean(desvio_padrao / media)

for name, data in results.items():
    results[name]["CV"] = coef_variacao(data["Modelo"], X_test)

# Exibir resultados
for name, metrics in results.items():
    print(f"{name} -> EQM: {metrics['EQM']:.6f}, EPMA: {metrics['EPMA']:.6f}, CV: {metrics['CV']:.6f}")

# Plotar evolução do EQM
plt.figure(figsize=(12, 6))
for name, metrics in results.items():
    plt.plot(metrics['Train_EQM'], label=f'{name} - Treino')
    plt.plot(metrics['Val_EQM'], label=f'{name} - Validação', linestyle='dashed')
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.title('Evolução do EQM Durante o Treinamento')
plt.legend()
plt.show()
