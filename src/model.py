import pandas as pd
import glob
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import os
import joblib

from etl import padronizar_texto

# ==========================================
# 1. VARRER A PASTA E CONSOLIDAR O CLIMA
# ==========================================

caminho_pasta = '/data/raw/DadosEstacoes/'
arquivos_meteo = glob.glob(caminho_pasta + 'dados_A*.csv')

lista_df = []

for arquivo in arquivos_meteo:
    try:
        # Abre o arquivo só para ler a primeira linha e capturar o nome da cidade
        with open(arquivo, 'r', encoding='utf-8') as f:
            linha_nome = f.readline()  # Ex: "Nome: BRASILIA"
            nome_cidade = linha_nome.split(':')[1].strip()

        # Agora lê os dados ignorando as 10 linhas de cabeçalho
        df_temp = pd.read_csv(arquivo, sep=';', skiprows=10, encoding='utf-8')

        # Adiciona a cidade identificada como uma nova coluna
        df_temp['Municipio_Clima'] = nome_cidade
        lista_df.append(df_temp)
    except Exception as e:
        print(f"Ignorando arquivo {arquivo} devido a erro: {e}")

# Junta todos os CSVs em uma tabela única
df_meteo = pd.concat(lista_df, ignore_index=True)

# Renomeia e limpa as colunas
df_meteo.columns = ['Data', 'DiasPrecip', 'PrecipTotal', 'PressaoMedia', 'TempMedia', 'VentoMax', 'VentoMedio', 'Vazia', 'Municipio_Clima']
df_meteo = df_meteo.drop(columns=['Vazia'])

# Padroniza datas e o nome da cidade para o cruzamento
df_meteo['Data'] = pd.to_datetime(df_meteo['Data'])
df_meteo['AnoMes'] = df_meteo['Data'].dt.to_period('M')
df_meteo['Chave_Cruzamento'] = df_meteo['Municipio_Clima'].apply(padronizar_texto)

# ==========================================
# 2. PREPARAR DADOS DO INPE
# ==========================================

inpe = pd.read_csv('data/raw/bdqueimadas_2025-09-07_2026-09-08.csv')

inpe['DataHora'] = pd.to_datetime(inpe['DataHora'])
inpe['AnoMes'] = inpe['DataHora'].dt.to_period('M')

# Padroniza o nome do município no INPE da mesma forma
inpe['Chave_Cruzamento'] = inpe['Municipio'].apply(padronizar_texto)

# Agrupa a contagem de focos de incêndio por Cidade e por Mês
focos_por_cidade = inpe.groupby(['Chave_Cruzamento', 'AnoMes']).size().reset_index(name='NumFocos')

# ==========================================
# 3. CRUZAMENTO (MERGE) E TREINO DA IA
# ==========================================

# Une as duas bases. Onde a data e a cidade baterem, os dados se juntam.
df_final = pd.merge(df_meteo, focos_por_cidade, on=['Chave_Cruzamento', 'AnoMes'], how='left')

# Se não teve foco no INPE para aquela cidade naquele mês, preenche com 0
df_final['NumFocos'] = df_final['NumFocos'].fillna(0)

# Remove linhas que não têm dado de temperatura ou chuva registrados
df_final = df_final.dropna(subset=['TempMedia', 'PrecipTotal'])

# Define o alvo: 1 se teve mais de 10 incêndios no mês, 0 caso contrário
df_final['RiscoAlto'] = (df_final['NumFocos'] > 10).astype(int)

# Treinamento do Modelo
X = df_final[['TempMedia', 'PrecipTotal']]
y = df_final['RiscoAlto']

modelo = LogisticRegression()
modelo.fit(X, y)

print("--- PIPELINE CONCLUÍDO ---")
print(f"Total de registros processados: {len(df_final)}")
print(f"Acurácia do modelo no treino: {modelo.score(X, y):.2%}")
print("Coeficientes (Pesos): Temperatura =", modelo.coef_[0][0], "| Chuva =", modelo.coef_[0][1])

# ==========================================
# 4. APLICAÇÃO DO XGBOOST
# ==========================================

# 1. Ajuste do Desbalanceamento
# Calcula a proporção exata entre os meses sem fogo (0) e com fogo (1) na sua base
contagem_zeros = (y == 0).sum()
contagem_uns = (y == 1).sum()
peso_classes = contagem_zeros / contagem_uns

# 2. Configuração e Treinamento do XGBoost
modelo_xgb = xgb.XGBClassifier(
    scale_pos_weight=peso_classes, # Força o modelo a dar mais atenção à minoria (incêndios)
    max_depth=4,                   # Limita a profundidade da árvore para evitar overfitting
    learning_rate=0.05,            # Taxa de aprendizado mais conservadora
    n_estimators=150,              # Quantidade de árvores de decisão geradas
    random_state=42,
    eval_metric='aucpr'            # Otimiza o modelo focado na curva Precision-Recall
)

modelo_xgb.fit(X, y)

# Cria uma pasta chamada 'modelos' na mesma pasta do script (caso ela não exista)
pasta_destino = "modelos"
os.makedirs(pasta_destino, exist_ok=True)

# Define o caminho completo do arquivo
nome_arquivo = os.path.join(pasta_destino, "modelo_risco_incendio_xgb.pkl")

# Salva o modelo
joblib.dump(modelo_xgb, nome_arquivo)

# Mostra o caminho absoluto onde foi salvo
print(f"Modelo salvo com sucesso em: {os.path.abspath(nome_arquivo)}")