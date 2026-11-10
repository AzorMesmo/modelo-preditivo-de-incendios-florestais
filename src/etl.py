import unicodedata
import pandas as pd
import numpy as np


def padronizar_texto(texto):
    if pd.isna(texto):
        return texto
    texto_sem_acento = ''.join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn')
    return texto_sem_acento.upper().strip()


def carregar_e_processar_dados(caminho_inmet: str, caminho_inpe: str) -> pd.DataFrame:
    try:
        df_inmet = pd.read_csv(caminho_inmet)
        df_inpe = pd.read_csv(caminho_inpe)
    except FileNotFoundError as e:
        print(f"Arquivo não encontrado: {e}")
        return pd.DataFrame()

    if 'DATA' in df_inmet.columns:
        df_inmet['DATA'] = pd.to_datetime(df_inmet['DATA'], errors='coerce')
    if 'DATA' in df_inpe.columns:
        df_inpe['DATA'] = pd.to_datetime(df_inpe['DATA'], errors='coerce')

    df_inmet.fillna(method='ffill', inplace=True)
    df_inpe.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    if 'MUNICIPIO' in df_inmet.columns:
        df_inmet['MUNICIPIO'] = df_inmet['MUNICIPIO'].apply(padronizar_texto)
    if 'MUNICIPIO' in df_inpe.columns:
        df_inpe['MUNICIPIO'] = df_inpe['MUNICIPIO'].apply(padronizar_texto)

    df_merged = pd.merge(df_inmet, df_inpe, on=['DATA', 'MUNICIPIO'], how='left', suffixes=('_inmet', '_inpe'))
    df_merged['RISCO_INCENDIO'] = df_merged['RISCO_INCENDIO'].fillna(0)

    return df_merged
