
#@title --- Preparar dados modelo LSTM saida multivariada ----- (15min - 27 Observações)


import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def filtrar_intervalo_horario(df):
    df['hora'] = pd.to_datetime(df['hora'], format='%H:%M:%S').dt.time
    hora_inicio = pd.to_datetime('10:05', format='%H:%M').time()
    hora_fim = pd.to_datetime('16:50', format='%H:%M').time()
    df = df[df['hora'].between(hora_inicio, hora_fim)]
    return df

def normalizar_padronizar_dados(ticker, path_dados_transformados, path_scalers, path_dados_normalizados):
    df = pd.read_csv(path_dados_transformados)
    df = df.sort_values(by=['data', 'hora']).reset_index(drop=True)
    df = filtrar_intervalo_horario(df)

    df = df[['data', 'hora', 'dia_semana', 'abertura', 'minimo', 'maximo',
             'fechamento', 'volume', 'retorno', 'volatilidade', 'hora_num', 'minuto']]

    df = pd.get_dummies(df, columns=['dia_semana'], prefix='dia_semana')

    colunas_normalizar = ['abertura', 'minimo', 'maximo', 'fechamento', 'volume', 'volatilidade', 'hora_num', 'minuto']
    colunas_padronizar = ['retorno']

    for col in colunas_normalizar + colunas_padronizar:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Coluna {col} não é numérica.")

    minmax_scaler = MinMaxScaler()
    df[colunas_normalizar] = minmax_scaler.fit_transform(df[colunas_normalizar])

    target_scaler = MinMaxScaler()
    df['fechamento'] = target_scaler.fit_transform(df[['fechamento']])

    standard_scaler = StandardScaler()
    df[colunas_padronizar] = standard_scaler.fit_transform(df[colunas_padronizar])

    os.makedirs(path_scalers, exist_ok=True)
    joblib.dump(minmax_scaler, os.path.join(path_scalers, 'minmax_scaler.pkl'))
    joblib.dump(standard_scaler, os.path.join(path_scalers, 'standard_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(path_scalers, 'target_scaler.pkl'))

    scalers_info = {
        'colunas_normalizadas': colunas_normalizar,
        'colunas_padronizadas': colunas_padronizar,
        'colunas_one_hot': [col for col in df.columns if col.startswith('dia_semana_')]
    }
    joblib.dump(scalers_info, os.path.join(path_scalers, 'scalers_info.pkl'))

    os.makedirs(path_dados_normalizados, exist_ok=True)
    df.to_csv(os.path.join(path_dados_normalizados, 'dados_normalizados.csv'), index=False)
    return df

def criar_sequencias(df, obs_dia, target_col, colunas_extra, path_sequencias):
    X, y, extras = [], [], []
    seq_len = 2 * obs_dia
    n_alvos = obs_dia

    for i in range(len(df) - seq_len - n_alvos + 1):
        seq = df.iloc[i : i + seq_len]
        alvo = df.iloc[i + seq_len : i + seq_len + n_alvos]

        if len(alvo['data'].unique()) != 1:
            continue

        seq_x = seq.drop(columns=['data', 'hora'] + colunas_extra).values.astype(np.float32)
        seq_y = alvo[target_col].values.astype(np.float32)
        X.append(seq_x)
        y.append(seq_y)
        extras.append(alvo.iloc[0][colunas_extra].values)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    extras = np.array(extras, dtype=object)

    os.makedirs(path_sequencias, exist_ok=True)
    np.save(os.path.join(path_sequencias, 'X.npy'), X)
    np.save(os.path.join(path_sequencias, 'y.npy'), y)
    np.save(os.path.join(path_sequencias, 'extras.npy'), extras)

    return X, y, extras

def dividir_dados(X, y, extras, path_treino_teste_val, proporcao_treino=0.7, proporcao_val=0.15):
    n = len(X)
    n_treino = int(n * proporcao_treino)
    n_val = int(n * proporcao_val)

    X_treino, y_treino, extras_treino = X[:n_treino], y[:n_treino], extras[:n_treino]
    X_val, y_val, extras_val = X[n_treino:n_treino + n_val], y[n_treino:n_treino + n_val], extras[n_treino:n_treino + n_val]
    X_teste, y_teste, extras_teste = X[n_treino + n_val:], y[n_treino + n_val:], extras[n_treino + n_val:]

    os.makedirs(path_treino_teste_val, exist_ok=True)
    np.save(os.path.join(path_treino_teste_val, 'X_treino.npy'), X_treino)
    np.save(os.path.join(path_treino_teste_val, 'y_treino.npy'), y_treino)
    np.save(os.path.join(path_treino_teste_val, 'extras_treino.npy'), extras_treino)

    np.save(os.path.join(path_treino_teste_val, 'X_val.npy'), X_val)
    np.save(os.path.join(path_treino_teste_val, 'y_val.npy'), y_val)
    np.save(os.path.join(path_treino_teste_val, 'extras_val.npy'), extras_val)

    np.save(os.path.join(path_treino_teste_val, 'X_teste.npy'), X_teste)
    np.save(os.path.join(path_treino_teste_val, 'y_teste.npy'), y_teste)
    np.save(os.path.join(path_treino_teste_val, 'extras_teste.npy'), extras_teste)

    return (X_treino, y_treino, extras_treino,
            X_val, y_val, extras_val,
            X_teste, y_teste, extras_teste)

def preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto):
    print(f"\n--- Preparando dados para o modelo LSTM para {ticker} na versão {versao_dados} ---")

    path_dados_transformados = os.path.join(path_projeto, f'data/transformed/{ticker}/{intervalo}/dados_transformados_{ticker}_{intervalo}.csv')
    base_dados = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}')
    path_dados_normalizados = os.path.join(base_dados, 'dados_normalizados')
    path_scalers = os.path.join(base_dados, 'scalers')
    path_sequencias = os.path.join(base_dados, 'sequencias')
    path_treino_teste_val = os.path.join(base_dados, 'treino_teste_val')

    print('\n[✔] Normalizando e padronizando dados...')
    df = normalizar_padronizar_dados(ticker, path_dados_transformados, path_scalers, path_dados_normalizados)

    print('\n[✔] Criando sequências de entrada e alvo...')
    obs_dia = 27
    target_col = 'fechamento'
    colunas_tempo = ['data', 'hora']

    X, y, extras = criar_sequencias(df, obs_dia, target_col, colunas_tempo, path_sequencias)

    print('\n[✔] Dividindo dados em treino, validação e teste...')
    dividir_dados(X, y, extras, path_treino_teste_val)

    print(f"\nOs dados normalizados estão em: {path_dados_normalizados}")
    print(f"Os scalers estão em: {path_scalers}")
    print(f"As sequências estão em: {path_sequencias}")
    print(f"Os conjuntos de treino, validação e teste estão em: {path_treino_teste_val}")

    return X, y, extras

if __name__ == "__main__":
    ticker = 'BBDC4'
    intervalo = '15m'
    versao_dados = 'v001'
    path_projeto = '/content/pregao_predict'
    preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto)
    print("\n[✔] Dados preparados com sucesso.")

