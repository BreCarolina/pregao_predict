#@title Preparar dados LSTM - Versão 02

'''
Normalização por pregão, criação de sequências e divisão dos dados.
'''

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados):
    df = pd.read_csv(path_dados_limpos)
    df['Data'] = pd.to_datetime(df['Data'])

    df = df[(df['Hora'] >= '10:10:00') & (df['Hora'] <= '16:45:00')]

    contagem_por_dia = df.groupby(df['Data'].dt.date).size()
    dias_validos = contagem_por_dia[contagem_por_dia == 80].index
    df = df[df['Data'].dt.date.isin(dias_validos)]

    print(f"\n[INFO] Filtrando apenas os pregões com 80 candles...")
    print(f"[INFO] {len(dias_validos)} pregões válidos encontrados.")

    df['Dia_semana'] = df['Data'].dt.day_name()
    df = pd.get_dummies(df, columns=['Dia_semana'], prefix='Dia_semana')

    colunas_temporais = ['Data', 'Hora'] + [col for col in df.columns if col.startswith('Dia_semana_')]
    colunas_normalizar = ['Abertura', 'Máximo', 'Mínimo']
    colunas_padronizar = ['Volume', 'Quantidade']
    coluna_target = 'Fechamento'

    X_sequencias = []
    y_sequencias = []
    scalers_por_dia = {}

    for idx in range(len(dias_validos) - 1):
        data_dia = dias_validos[idx]
        data_proximo = dias_validos[idx + 1]

        df_dia = df[df['Data'].dt.date == data_dia].copy()
        df_proximo = df[df['Data'].dt.date == data_proximo].copy()

        if df_dia.shape[0] != 80 or df_proximo.shape[0] != 80:
            print(f"[AVISO] Ignorando dias com quantidade de candles diferente de 80: {data_dia}, {data_proximo}")
            continue

        scaler_preco = MinMaxScaler()
        scaler_target = MinMaxScaler()
        scaler_volume = StandardScaler()

        df_dia[colunas_normalizar] = scaler_preco.fit_transform(df_dia[colunas_normalizar])
        df_dia[colunas_padronizar] = scaler_volume.fit_transform(df_dia[colunas_padronizar])
        df_dia[[coluna_target]] = scaler_target.fit_transform(df_dia[[coluna_target]])

        scaler_target_y = MinMaxScaler()
        df_proximo_y = scaler_target_y.fit_transform(df_proximo[[coluna_target]])

        X_dia = df_dia.drop(columns=colunas_temporais).values
        y_dia = df_proximo_y.flatten()

        if X_dia.shape != (80, X_dia.shape[1]) or y_dia.shape != (80,):
            print(f"[ERRO] Formato inesperado ao processar {data_dia}. Ignorando.")
            continue

        print(f"[OK] Pregão {data_dia} -> {data_proximo} | X: {X_dia.shape}, y: {y_dia.shape}")

        X_sequencias.append(X_dia)
        y_sequencias.append(y_dia)

        scalers_por_dia[str(data_dia)] = {
            'scaler_preco': scaler_preco,
            'scaler_volume': scaler_volume,
            'scaler_target_X': scaler_target,
            'scaler_target_y': scaler_target_y
        }

    X = np.array(X_sequencias)
    y = np.array(y_sequencias)

    print(f"\n[INFO] Formato final das sequências: X={X.shape}, y={y.shape}")

    os.makedirs(path_normalizados, exist_ok=True)
    os.makedirs(path_scalers, exist_ok=True)
    np.save(os.path.join(path_normalizados, 'X.npy'), X)
    np.save(os.path.join(path_normalizados, 'y.npy'), y)
    joblib.dump(scalers_por_dia, os.path.join(path_scalers, 'scalers_por_pregao.pkl'))

    return X, y, scalers_por_dia

def dividir_e_salvar_dados(X, y, path_treino_val_teste, proporcao_val=0.15, proporcao_teste=0.15):
    os.makedirs(path_treino_val_teste, exist_ok=True)

    X_temp, X_teste, y_temp, y_teste = train_test_split(X, y, test_size=proporcao_teste, shuffle=False)
    X_treino, X_val, y_treino, y_val = train_test_split(X_temp, y_temp, test_size=proporcao_val / (1 - proporcao_teste), shuffle=False)

    np.save(os.path.join(path_treino_val_teste, 'X_treino.npy'), X_treino)
    np.save(os.path.join(path_treino_val_teste, 'y_treino.npy'), y_treino)
    np.save(os.path.join(path_treino_val_teste, 'X_val.npy'), X_val)
    np.save(os.path.join(path_treino_val_teste, 'y_val.npy'), y_val)
    np.save(os.path.join(path_treino_val_teste, 'X_teste.npy'), X_teste)
    np.save(os.path.join(path_treino_val_teste, 'y_teste.npy'), y_teste)

    print(f"\n[✓] Divisão concluída:")
    print(f"  -> Treino:      {X_treino.shape[0]} sequências")
    print(f"  -> Validação:   {X_val.shape[0]} sequências")
    print(f"  -> Teste:       {X_teste.shape[0]} sequências")

def preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto):
    print(f"\n[INICIO] Preparando dados LSTM para {ticker} - Versao: {versao_dados}")

    path_dados_limpos = os.path.join(path_projeto, f"data/cleaned/{ticker}/{intervalo}/dados_limpos_{ticker}_{intervalo}.csv")
    path_dados_preparados = os.path.join(path_projeto, f"data/prepared/{ticker}/{intervalo}_{versao_dados}")
    path_normalizados = os.path.join(path_dados_preparados, 'normalizados')
    path_scalers = os.path.join(path_dados_preparados, 'scalers')
    path_treino_val_teste = os.path.join(path_dados_preparados, 'treino_teste_val')

    X, y, scalers_por_dia = normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados)
    dividir_e_salvar_dados(X, y, path_treino_val_teste)

    print("[✓] Preparação dos dados concluída.")

if __name__ == "__main__":
    preparar_dados_lstm(
        ticker='LREN3',
        intervalo='5m',
        versao_dados='v2',
        path_projeto='/content/pregao_predict'
    )
