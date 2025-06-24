
# @title Script de transformação de dados

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

# Suprime alertas do tipo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Verifica a última data dos dados já transformados
def verificar_dados_existentes(path_dados_transformados):    
    data_inicio = None
    if os.path.exists(path_dados_transformados):
        df_transformado_existente = pd.read_csv(path_dados_transformados)
        df_transformado_existente["data"] = pd.to_datetime(df_transformado_existente["data"], errors="coerce")
        if "data" in df_transformado_existente.columns and pd.notnull(df_transformado_existente["data"].max()):
            ultimo_registro = df_transformado_existente["data"].max().date()
            print(f"Último registro transformado encontrado: {ultimo_registro}")
            data_inicio = ultimo_registro + timedelta(days=1)
    else:
        df_transformado_existente = pd.DataFrame()
        print("Nenhum dado transformado encontrado.")
        data_inicio = pd.to_datetime("1900-01-01").date()  # Valor padrão para iniciar do início

    return df_transformado_existente, data_inicio

# Carrega os dados limpos e filtra os que ainda não foram transformados
def carregar_dados_limpos(path_dados_limpos: str, data_inicio) -> pd.DataFrame:
    df_limpo = pd.read_csv(path_dados_limpos, parse_dates=['data', 'datetime'])
    df_transformar = df_limpo[df_limpo['data'].dt.date >= data_inicio].copy()
    return df_transformar

# Adiciona features de tempo (hora, minuto, dia da semana)
def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    print("Adicionando features temporais...")
    df['hora_num'] = df['datetime'].dt.hour
    df['minuto'] = df['datetime'].dt.minute
    df['dia_semana'] = df['datetime'].dt.dayofweek
    print("Features temporais adicionadas.")
    return df

# Calcula o retorno percentual entre candles consecutivos
def calcular_retorno(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculando retorno percentual...")
    df['retorno'] = df['fechamento'].pct_change().fillna(0)
    return df

# Calcula a volatilidade como desvio padrão móvel do retorno
def calcular_volatilidade(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculando volatilidade...")
    df['volatilidade'] = df['retorno'].rolling(window=10, min_periods=1).std().fillna(0)
    return df

# Salva os dois últimos pregões em CSV separado
def salvar_ultimos_pregoes(df: pd.DataFrame, ticker: str, path_daily: str) -> None:
    df['data'] = pd.to_datetime(df['data'])
    datas_recentes = sorted(df['data'].dt.date.unique())[-2:]
    df_daily = df[df['data'].dt.date.isin(datas_recentes)].copy()
    caminho_saida = os.path.join(path_daily, f"dois_ultimos_pregoes.csv")
    df_daily.to_csv(caminho_saida, index=False)
    print(f"[✓] Daily salvo em: {caminho_saida}")

# Pipeline principal de transformação
def transformar_dados(ticker: str, intervalo: str, path_projeto: str) -> None:
    print(f"\n--- Transformação de dados iniciada ---\n")

    # Define caminhos
    path_dados_limpos = os.path.join(path_projeto, f'data/cleaned/{ticker.split(".")[0]}/dados_limpos_{ticker.split(".")[0]}_{intervalo}.csv')
    path_dados_transformados = os.path.join(path_projeto, f'data/transformed/{ticker.split(".")[0]}/{intervalo}')
    path_arquivo_transformado = os.path.join(path_dados_transformados, f'dados_transformados_{ticker.split(".")[0]}_{intervalo}.csv')
    path_daily = os.path.join(path_dados_transformados, f'daily_{intervalo}')

    os.makedirs(path_dados_transformados, exist_ok=True)
    os.makedirs(path_daily, exist_ok=True)

    # Verifica transformados anteriores e data inicial
    df_transformado_existente, data_inicio = verificar_dados_existentes(path_arquivo_transformado)

    # Carrega dados limpos a transformar
    df_transformar = carregar_dados_limpos(path_dados_limpos, data_inicio)
    if df_transformar.empty:
        print("[AVISO] Nenhum dado novo para transformar. Encerrando.")
        return

    # Aplica transformações
    df_transformar = adicionar_features_temporais(df_transformar)
    df_transformar = calcular_retorno(df_transformar)
    df_transformar = calcular_volatilidade(df_transformar)

    # Junta com histórico transformado, se houver
    df_final = pd.concat([df_transformado_existente, df_transformar], ignore_index=True)

    # Define colunas finais
    colunas_finais = [
        'data', 'hora',
        'abertura', 'minimo', 'maximo', 'fechamento', 'volume',
        'retorno', 'volatilidade', 'dia_semana', 'hora_num', 'minuto', 'datetime'
    ]
    for col in colunas_finais:
        if col not in df_final.columns:
            df_final[col] = np.nan
    df_final = df_final[colunas_finais]
    df_final.sort_values(by='datetime', inplace=True)

    # Salva o resultado completo
    df_final.to_csv(path_arquivo_transformado, index=False)
    print(f"[✓] Transformação concluída. Dados salvos em: {path_arquivo_transformado}. Total de {len(df_final)} registros.")

    # Salva últimos dois pregões
    salvar_ultimos_pregoes(df_final, ticker, path_daily)

# Execução para testes
if __name__ == "__main__":
    ticker = 'BBDC4'
    intervalo = '15m'
    path_projeto = "/content/pregao_predict"
    transformar_dados(ticker, intervalo, path_projeto)
