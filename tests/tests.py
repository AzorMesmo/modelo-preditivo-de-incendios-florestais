import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import xgboost as xgb

from src.model import X, y, modelo, modelo_xgb

# 1. Extração das Probabilidades Matemáticas do modelo
y_prob_classe1 = modelo.predict_proba(X)[:, 1]

# Garante que 'y' seja um array numpy para evitar problemas de indexação com o Pandas
y_array = np.array(y)

# 2. Configuração da Figura
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- GRÁFICO 1: CURVA ROC ---
fpr, tpr, thresholds_roc = roc_curve(y_array, y_prob_classe1)
roc_auc = auc(fpr, tpr)

axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('Taxa de Falsos Positivos')
axes[0].set_ylabel('Taxa de Verdadeiros Positivos')
axes[0].set_title('Curva ROC')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# --- GRÁFICO 2: CURVA PRECISION-RECALL ---
precision, recall, thresholds_pr = precision_recall_curve(y_array, y_prob_classe1)
pr_auc = average_precision_score(y_array, y_prob_classe1)

axes[1].plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
axes[1].set_xlabel('Recall (Revocação - % de focos detectados)')
axes[1].set_ylabel('Precision (Precisão - % de alertas corretos)')
axes[1].set_title('Curva Precision-Recall')
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

# --- GRÁFICO 3: CURVA DE GANHO (CUMULATIVE GAINS) NATIVA ---
# Ordena as predições da maior probabilidade para a menor
sorted_indices = np.argsort(y_prob_classe1)[::-1]
y_true_sorted = y_array[sorted_indices]

# Calcula as porcentagens acumuladas
cum_true = np.cumsum(y_true_sorted)
gain = cum_true / np.sum(y_array)
percentages = np.arange(1, len(y_array) + 1) / len(y_array)

# Adiciona o ponto de origem (0,0) no gráfico
percentages = np.insert(percentages, 0, 0)
gain = np.insert(gain, 0, 0)

axes[2].plot(percentages, gain, color='purple', lw=2, label='Modelo Logístico')
axes[2].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Aleatório (Baseline)')
axes[2].set_title('Curva de Ganho Cumulativo')
axes[2].set_xlabel('% da Amostra Analisada')
axes[2].set_ylabel('% de Focos Encontrados')
axes[2].legend(loc="lower right")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 1. Extração das Probabilidades do novo modelo
y_prob_xgb = modelo_xgb.predict_proba(X)[:, 1]
y_array = np.array(y)

# 2. Renderização das Curvas de Avaliação
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- GRÁFICO 1: CURVA ROC ---
fpr, tpr, _ = roc_curve(y_array, y_prob_xgb)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_title('XGBoost: Curva ROC')
axes[0].set_xlabel('Taxa de Falsos Positivos')
axes[0].set_ylabel('Taxa de Verdadeiros Positivos')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# --- GRÁFICO 2: CURVA PRECISION-RECALL ---
precision, recall, _ = precision_recall_curve(y_array, y_prob_xgb)
pr_auc = average_precision_score(y_array, y_prob_xgb)
axes[1].plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
axes[1].set_title('XGBoost: Curva Precision-Recall')
axes[1].set_xlabel('Recall (Revocação)')
axes[1].set_ylabel('Precision (Precisão)')
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

# --- GRÁFICO 3: CURVA DE GANHO CUMULATIVO ---
sorted_indices = np.argsort(y_prob_xgb)[::-1]
y_true_sorted = y_array[sorted_indices]
cum_true = np.cumsum(y_true_sorted)
gain = cum_true / np.sum(y_array)
percentages = np.arange(1, len(y_array) + 1) / len(y_array)
percentages = np.insert(percentages, 0, 0)
gain = np.insert(gain, 0, 0)

axes[2].plot(percentages, gain, color='purple', lw=2, label='Modelo XGBoost')
axes[2].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Aleatório (Baseline)')
axes[2].set_title('XGBoost: Curva de Ganho Cumulativo')
axes[2].set_xlabel('% da Amostra Analisada')
axes[2].set_ylabel('% de Focos Encontrados')
axes[2].legend(loc="lower right")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Exibe a importância de cada variável na decisão do algoritmo
xgb.plot_importance(modelo_xgb, importance_type='weight', title='Importância das Variáveis (Temperatura vs Chuva)')
plt.show()
