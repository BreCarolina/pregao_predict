#@title Preparar dados LSTM - Normalização por pregão, criação de sequências e divisão dos dados Versão 04

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normaliza cada pregão separadamente

def normalizar_por_pregao(df, colunas_normalizar, colunas_padronizar, coluna_target):
    df_normalizado = []
    grouped = df.groupby(df['Data'].dt.date)

    for data, grupo in grouped:
        grupo = grupo.copy()
        scaler_preco = MinMaxScaler()
        scaler_target = MinMaxScaler()
        scaler_volume = StandardScaler()

        grupo[colunas_normalizar] = scaler_preco.fit_transform(grupo[colunas_normalizar])
        grupo[[coluna_target]] = scaler_target.fit_transform(grupo[[coluna_target]])
        grupo[colunas_padronizar] = scaler_volume.fit_transform(grupo[colunas_padronizar])

        grupo['Scaler_preco'] = [scaler_preco] * len(grupo)
        grupo['Scaler_target'] = [scaler_target] * len(grupo)
        grupo['Scaler_volume'] = [scaler_volume] * len(grupo)

        df_normalizado.append(grupo)

    return pd.concat(df_normalizado, ignore_index=True)


def normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados):
    df = pd.read_csv(path_dados_limpos)
    df['Data'] = pd.to_datetime(df['Data'])

    df = df[(df['Hora'] >= '10:10:00') & (df['Hora'] <= '16:45:00')]
    contagem_por_dia = df.groupby(df['Data'].dt.date).size()
    dias_validos = contagem_por_dia[contagem_por_dia == 80].index
    df = df[df['Data'].dt.date.isin(dias_validos)]

    df['Dia_semana'] = df['Data'].dt.day_name()
    df = pd.get_dummies(df, columns=['Dia_semana'], prefix='Dia_semana')

    colunas_temporais = ['Data', 'Hora'] + [col for col in df.columns if col.startswith('Dia_semana_')]
    colunas_normalizar = ['Abertura', 'Máximo', 'Mínimo']
    colunas_padronizar = ['Volume', 'Quantidade']
    coluna_target = 'Fechamento'

    df = normalizar_por_pregao(df, colunas_normalizar, colunas_padronizar, coluna_target)

    os.makedirs(path_scalers, exist_ok=True)
    joblib.dump({
        'colunas_normalizadas': colunas_normalizar,
        'colunas_padronizadas': colunas_padronizar,
        'colunas_dummy': [col for col in df.columns if col.startswith('Dia_semana_')],
        'target_col': coluna_target
    }, os.path.join(path_scalers, 'scalers_info.pkl'))

    os.makedirs(path_normalizados, exist_ok=True)
    df.to_csv(os.path.join(path_normalizados, 'dados_normalizados.csv'), index=False)

    return df, colunas_temporais


def criar_sequencias(df, path_sequencias):
    print("[INFO] Criando sequencias com janela por pregao...")
    X, y, tempos = [], [], []

    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by=['Data', 'Hora']).reset_index(drop=True)
    dias_unicos = df['Data'].dt.date.unique()

    colunas_features = [col for col in df.columns if col not in ['Data', 'Hora', 'Ativo', 'Scaler_preco', 'Scaler_target', 'Scaler_volume']]
    grouped = df.groupby(df['Data'].dt.date)

    for i in range(len(dias_unicos) - 1):
        dia1 = dias_unicos[i]

        if pd.to_datetime(dia1).weekday() == 4 and i + 2 < len(dias_unicos):
            dia2 = dias_unicos[i + 2]
        else:
            dia2 = dias_unicos[i + 1]

        df1 = grouped.get_group(dia1)[colunas_features]
        df2 = grouped.get_group(dia2)['Fechamento']

        n = min(len(df1), len(df2))
        if n < 10:
            continue

        entrada = df1.iloc[:n].values.astype(np.float32)
        saida = df2.iloc[:n].values.astype(np.float32)

        if np.isnan(entrada).any() or np.isnan(saida).any():
            continue

        X.append(entrada)
        y.append(saida)
        tempos.append(dia2)

    X = np.array(X)
    y = np.array(y)
    tempos = np.array(tempos, dtype=object)

    os.makedirs(path_sequencias, exist_ok=True)
    np.save(os.path.join(path_sequencias, 'X.npy'), X)
    np.save(os.path.join(path_sequencias, 'y.npy'), y)
    np.save(os.path.join(path_sequencias, 'tempos.npy'), tempos)

    print(f"[INFO] Sequencias geradas: {X.shape[0]} exemplos.")
    return X, y, tempos


def dividir_dados(X, y, tempos, path_divisao, proporcao_treino=0.7, proporcao_val=0.15):
    total = len(X)
    n_treino = int(total * proporcao_treino)
    n_val = int(total * proporcao_val)

    X_treino, y_treino, t_treino = X[:n_treino], y[:n_treino], tempos[:n_treino]
    X_val, y_val, t_val = X[n_treino:n_treino + n_val], y[n_treino:n_treino + n_val], tempos[n_treino:n_treino + n_val]
    X_teste, y_teste, t_teste = X[n_treino + n_val:], y[n_treino + n_val:], tempos[n_treino + n_val:]

    os.makedirs(path_divisao, exist_ok=True)
    np.save(os.path.join(path_divisao, 'X_treino.npy'), X_treino)
    np.save(os.path.join(path_divisao, 'y_treino.npy'), y_treino)
    np.save(os.path.join(path_divisao, 'tempos_treino.npy'), t_treino)
    np.save(os.path.join(path_divisao, 'X_val.npy'), X_val)
    np.save(os.path.join(path_divisao, 'y_val.npy'), y_val)
    np.save(os.path.join(path_divisao, 'tempos_val.npy'), t_val)
    np.save(os.path.join(path_divisao, 'X_teste.npy'), X_teste)
    np.save(os.path.join(path_divisao, 'y_teste.npy'), y_teste)
    np.save(os.path.join(path_divisao, 'tempos_teste.npy'), t_teste)

    print(f"[INFO] Divisao: treino={len(X_treino)}, val={len(X_val)}, teste={len(X_teste)}")
    return X_treino, y_treino, t_treino, X_val, y_val, t_val, X_teste, y_teste, t_teste


def preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto):
    print(f"\n[INICIO] Preparando dados LSTM para {ticker} - Versao: {versao_dados}")

    base_path = os.path.join(path_projeto, 'data')
    path_dados_limpos = os.path.join(base_path, f'cleaned/{ticker}/{intervalo}/dados_limpos_{ticker}_{intervalo}.csv')
    path_dados_preparados = os.path.join(base_path, f'prepared/{ticker}/{intervalo}_{versao_dados}')

    path_normalizados = os.path.join(path_dados_preparados, 'dados_normalizados')
    path_scalers = os.path.join(path_dados_preparados, 'scalers')
    path_sequencias = os.path.join(path_dados_preparados, 'sequencias')
    path_divisao = os.path.join(path_dados_preparados, 'treino_teste_val')

    df, colunas_temporais = normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados)
    X, y, tempos = criar_sequencias(df, path_sequencias)

    if X.shape[0] == 0:
        print("[AVISO] Nenhuma sequencia foi criada. Verifique os dados.")
        return None, None, None

    dividir_dados(X, y, tempos, path_divisao)
    print("\n[\u2713] Pipeline finalizado com sucesso!")
    return X, y, tempos


def main():
    preparar_dados_lstm(
        ticker='LREN3',
        intervalo='5m',
        versao_dados='v4',
        path_projeto='/content/pregao_predict'
    )

if __name__ == "__main__":
    main()
