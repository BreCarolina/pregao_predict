
# @title Script de transformação de dados

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

# Suprime alertas do tipo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Verifica a última data transformada, se houver
def verificar_dados_existentes(path_dados_transformados):
    if os.path.exists(path_dados_transformados):
        df_existente = pd.read_csv(path_dados_transformados, parse_dates=['data'])
        ultima_data = df_existente["data"].max().date()
        print(f"[INFO] Último registro transformado encontrado: {ultima_data}")
        data_inicio = ultima_data + timedelta(days=1)
    else:
        df_existente = pd.DataFrame()
        print("[INFO] Nenhum dado transformado encontrado.")
        data_inicio = pd.to_datetime("1900-01-01").date()

    return df_existente, data_inicio

# Carrega dados limpos e filtra apenas os que ainda não foram transformados
def carregar_dados_limpos(path_dados_limpos: str, data_inicio) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_dados_limpos, parse_dates=['data', 'datetime'])
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {path_dados_limpos}")
        return None

    df_novos = df[df['data'].dt.date >= data_inicio].copy()

    if df_novos.empty:
        print("[AVISO] Nenhum dado novo para transformar. Encerrando.")
        return None

    return df_novos

# Adiciona colunas com informações temporais
def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Adicionando features temporais...")
    df['hora_num'] = df['datetime'].dt.hour
    df['minuto'] = df['datetime'].dt.minute
    df['dia_semana'] = df['datetime'].dt.dayofweek
    return df

# Calcula o retorno percentual entre candles
def calcular_retorno(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Calculando retorno percentual...")
    df['retorno'] = df['fechamento'].pct_change().fillna(0)
    return df

# Calcula volatilidade como desvio padrão móvel do retorno
def calcular_volatilidade(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Calculando volatilidade (desvio padrão)...")
    df['volatilidade'] = df['retorno'].rolling(window=10, min_periods=1).std().fillna(0)
    return df

# Salva os dois últimos pregões em CSV separado
def salvar_ultimos_pregoes(df: pd.DataFrame, ticker: str, intervalo: str, path_saida: str) -> None:
    df['data'] = pd.to_datetime(df['data'])
    datas_recentes = sorted(df['data'].dt.date.unique())[-2:]
    df_ultimos = df[df['data'].dt.date.isin(datas_recentes)].copy()
    caminho_saida = os.path.join(path_saida, f'dois_ultimos_pregoes_{ticker.split(".")[0]}_{intervalo}.csv')
    df_ultimos.to_csv(caminho_saida, index=False)
    print(f"[✓] Arquivo dos dois últimos pregões salvo em: {caminho_saida}")

# Pipeline principal
def transformar_dados(ticker: str, intervalo: str, path_projeto: str) -> None:
    print(f"\n--- Iniciando transformação de dados para {ticker.split('.')[0]} ({intervalo}) ---\n")

    # Define caminhos
    path_dados_limpos = os.path.join(path_projeto, f'data/cleaned/{ticker.split(".")[0]}/dados_limpos_{ticker.split(".")[0]}_{intervalo}.csv')
    path_dados_transformados = os.path.join(path_projeto, f'data/transformed/{ticker.split(".")[0]}/{intervalo}')
    path_arquivo_transformado = os.path.join(path_dados_transformados, f'dados_transformados_{ticker.split(".")[0]}_{intervalo}.csv')

    os.makedirs(path_dados_transformados, exist_ok=True)

    # Verifica dados anteriores e data de início
    df_existente, data_inicio = verificar_dados_existentes(path_arquivo_transformado)

    # Carrega dados limpos a transformar
    df_transformar = carregar_dados_limpos(path_dados_limpos, data_inicio)
    if df_transformar is None:
        return  # Encerra o script silenciosamente

    # Aplica transformações
    df_transformar = adicionar_features_temporais(df_transformar)
    df_transformar = calcular_retorno(df_transformar)
    df_transformar = calcular_volatilidade(df_transformar)

    # Junta com transformado anterior, se houver
    df_final = pd.concat([df_existente, df_transformar], ignore_index=True)

    # Define e garante presença das colunas finais
    colunas_finais = [
        'data', 'hora', 'abertura', 'minimo', 'maximo', 'fechamento', 'volume',
        'retorno', 'volatilidade', 'dia_semana', 'hora_num', 'minuto', 'datetime'
    ]
    for col in colunas_finais:
        if col not in df_final.columns:
            df_final[col] = np.nan

    df_final = df_final[colunas_finais]
    df_final.sort_values(by='datetime', inplace=True)

    # Salva dados transformados completos
    df_final.to_csv(path_arquivo_transformado, index=False)
    print(f"[✓] Transformação concluída. Total de {len(df_final)} registros salvos em: {path_arquivo_transformado}")

    # Salva os dois últimos pregões
    salvar_ultimos_pregoes(df_final, ticker, intervalo, path_dados_transformados)

# Execução direta para testes
if __name__ == "__main__":
    ticker = 'BBDC4'
    intervalo = '15m'
    path_projeto = "/content/pregao_predict"
    transformar_dados(ticker, intervalo, path_projeto)
