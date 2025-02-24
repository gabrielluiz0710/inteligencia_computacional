import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from ucimlrepo import fetch_ucirepo

# Fetch dataset
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

# Normalização conforme a fórmula do artigo
def normalize_data(X):
    L_min, L_max = 0, 1
    V_min, V_max = X.min(), X.max()
    return ((L_max - L_min) * (X - V_min) / (V_max - V_min)) + L_min

X = normalize_data(X)
y = normalize_data(y)

# Divisão dos dados conforme o artigo
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.233, random_state=42)

# Definição das arquiteturas
architectures = [[13, 13, 1], [13, 13, 13, 1], [13, 13, 13, 13, 1]]
learning_rate = 0.1
epochs = 5000

# Função para calcular métricas
def calculate_metrics(y_true, y_pred):
    EQM = np.mean((y_true - y_pred) ** 2)
    valid_indices = y_true != 0
    EPMA = np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / y_true[valid_indices]))
    sigma = np.std(y_pred)
    x_bar = np.mean(y_pred)
    coef_variacao = sigma / x_bar
    return EQM, EPMA, coef_variacao

# Função para treinar e avaliar a RNA
def train_and_evaluate(topology):
    mlp = MLPRegressor(hidden_layer_sizes=topology[1:-1], activation='logistic', solver='adam',
                        learning_rate_init=learning_rate, max_iter=epochs, random_state=42)
    
    mlp.fit(X_train, y_train.values.ravel())
    y_pred = mlp.predict(X_test)
    
    EQM, EPMA, coef_variacao = calculate_metrics(y_test.values.ravel(), y_pred)
    
    return EQM, EPMA, coef_variacao, mlp.loss_curve_

# Avaliação das arquiteturas
results = {}
plt.figure(figsize=(10, 6))
for topology in architectures:
    EQM, EPMA, coef_variacao, loss_curve = train_and_evaluate(topology)
    results[str(topology)] = {'EQM': EQM, 'EPMA': EPMA, 'Coef_Variacao': coef_variacao}
    plt.plot(loss_curve, label=f"{topology}")

# Gráficos conforme o artigo
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático Médio (EQM)")
plt.title("Evolução do EQM durante o Treinamento")
plt.legend()
plt.show()

# Exibir resultados formatados
print("Resultados das Arquiteturas:")
for arch, res in results.items():
    print(f"Arquitetura {arch}: EQM={res['EQM']:.5f}, EPMA={res['EPMA']:.5f}, Coef_Variacao={res['Coef_Variacao']:.5f}")
