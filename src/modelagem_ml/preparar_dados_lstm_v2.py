
#@title Preparar dados LSTM - Versão 02

'''
Normalização por pregão, criação de sequências e divisão dos dados 
'''

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Função para normalizar os dados por pregão e salvar os scalers por dia

def normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados):
    df = pd.read_csv(path_dados_limpos)
    df['Data'] = pd.to_datetime(df['Data'])

    # Filtra o horário do pregão entre 10:10 e 16:45
    df = df[(df['Hora'] >= '10:10:00') & (df['Hora'] <= '16:45:00')]

    # Conta candles por dia e filtra apenas os dias com 80 candles
    contagem_por_dia = df.groupby(df['Data'].dt.date).size()
    dias_validos = contagem_por_dia[contagem_por_dia == 80].index
    df = df[df['Data'].dt.date.isin(dias_validos)]

    print(f"\n[INFO] Filtrando apenas os pregões com 80 candles...")
    print(f"[INFO] {len(dias_validos)} pregões válidos encontrados.")

    # Cria variáveis categóricas para o dia da semana
    df['Dia_semana'] = df['Data'].dt.day_name()
    df = pd.get_dummies(df, columns=['Dia_semana'], prefix='Dia_semana')

    # Define colunas
    colunas_temporais = ['Data', 'Hora'] + [col for col in df.columns if col.startswith('Dia_semana_')]
    colunas_normalizar = ['Abertura', 'Máximo', 'Mínimo']
    colunas_padronizar = ['Volume', 'Quantidade']
    coluna_target = 'Fechamento'

    # Inicializa armazenamento de resultados
    X_sequencias = []
    y_sequencias = []
    scalers_por_dia = {}

    # Itera sobre pregões
    for idx in range(len(dias_validos) - 1):
        data_dia = dias_validos[idx]
        data_proximo = dias_validos[idx + 1]

        df_dia = df[df['Data'].dt.date == data_dia].copy()
        df_proximo = df[df['Data'].dt.date == data_proximo].copy()

        # Cria scalers e normaliza o dia atual
        scaler_preco = MinMaxScaler()
        scaler_target = MinMaxScaler()
        scaler_volume = StandardScaler()

        df_dia[colunas_normalizar] = scaler_preco.fit_transform(df_dia[colunas_normalizar])
        df_dia[colunas_padronizar] = scaler_volume.fit_transform(df_dia[colunas_padronizar])
        df_dia[[coluna_target]] = scaler_target.fit_transform(df_dia[[coluna_target]])

        # Normaliza target do próximo pregão
        scaler_target_y = MinMaxScaler()
        df_proximo_y = scaler_target_y.fit_transform(df_proximo[[coluna_target]])

        # Salva sequência de entrada e target
        X_sequencias.append(df_dia.drop(columns=colunas_temporais).values)
        y_sequencias.append(df_proximo_y.flatten())

        # Guarda scalers por data
        scalers_por_dia[str(data_dia)] = {
            'scaler_preco': scaler_preco,
            'scaler_volume': scaler_volume,
            'scaler_target_X': scaler_target,
            'scaler_target_y': scaler_target_y
        }

    # Converte para arrays numpy
    X = np.array(X_sequencias)
    y = np.array(y_sequencias)

    print(f"\n[INFO] Formato final das sequências: X={X.shape}, y={y.shape}")

    # Salva dados e scalers
    os.makedirs(path_normalizados, exist_ok=True)
    os.makedirs(path_scalers, exist_ok=True)
    np.save(os.path.join(path_normalizados, 'X.npy'), X)
    np.save(os.path.join(path_normalizados, 'y.npy'), y)
    joblib.dump(scalers_por_dia, os.path.join(path_scalers, 'scalers_por_pregao.pkl'))

    return X, y, scalers_por_dia

# Execução direta

def preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto):
    print(f"\n[INICIO] Preparando dados LSTM para {ticker} - Versao: {versao_dados}")

    path_dados_limpos = os.path.join(path_projeto, f"data/clean/{ticker}/{intervalo}_{versao_dados}.csv")
    path_dados_preparados = os.path.join(path_projeto, f"data/prepared/{ticker}/{intervalo}_{versao_dados}")
    path_normalizados = os.path.join(path_dados_preparados, 'normalizados')
    path_scalers = os.path.join(path_dados_preparados, 'scalers')

    X, y, scalers_por_dia = normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados)

    print("[✓] Preparação dos dados concluída.")

if __name__ == "__main__":
    preparar_dados_lstm(
        ticker='LREN3',
        intervalo='5m',
        versao_dados='v2',
        path_projeto='/content/pregao_predict'
    )
