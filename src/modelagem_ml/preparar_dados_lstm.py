#@title SRC Transformação de dados para LSTM

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preencher_dias_incompletos(df, obs_por_dia):
    print("[INFO] Preenchendo dias incompletos...")
    df['Data'] = pd.to_datetime(df['Data'])
    dias = df['Data'].dt.date.unique()
    df_preenchido = []

    for dia in dias:
        df_dia = df[df['Data'].dt.date == dia]
        if len(df_dia) < obs_por_dia:
            faltam = obs_por_dia - len(df_dia)
            ultimo = df_dia.iloc[-1].copy()
            for _ in range(faltam):
                novo = ultimo.copy()
                novo['Hora'] = pd.NaT
                # Se precisar preencher outras colunas não numéricas, ajustar aqui
                df_dia = pd.concat([df_dia, pd.DataFrame([novo])], ignore_index=True)
        df_preenchido.append(df_dia)

    df_final = pd.concat(df_preenchido).reset_index(drop=True)
    print("[INFO] Preenchimento concluído.")
    return df_final

def normalizar_padronizar_dados(ticker, path_dados_limpos, path_scalers, path_normalizados):
    df = pd.read_csv(path_dados_limpos)
    df['Data'] = pd.to_datetime(df['Data'])

    obs_por_dia = df.groupby(df['Data']).size().mode()[0]
    print(f"[INFO] Observações por dia mais frequente: {obs_por_dia}")

    print("\n[INFO] Criando variáveis categóricas para dias da semana...")
    df['Dia_semana'] = df['Data'].dt.day_name()
    df = pd.get_dummies(df, columns=['Dia_semana'], prefix='Dia_semana')
    print("[INFO] Variáveis categóricas criadas para dias da semana.")

    df = preencher_dias_incompletos(df, obs_por_dia)

    colunas_temporais = ['Data', 'Hora']
    colunas_normalizar = ['Abertura', 'Máximo', 'Mínimo']
    colunas_padronizar = ['Volume', 'Quantidade']
    coluna_target = 'Fechamento'

    print("\n[INFO] Normalizando colunas de preço...")
    minmax_scaler = MinMaxScaler()
    df[colunas_normalizar] = minmax_scaler.fit_transform(df[colunas_normalizar])
    print(f"[INFO] Colunas normalizadas (MinMaxScaler): {colunas_normalizar}")

    print("\n[INFO] Normalizando coluna target separadamente...")
    target_scaler = MinMaxScaler()
    df[[coluna_target]] = target_scaler.fit_transform(df[[coluna_target]])
    print(f"[INFO] Coluna target '{coluna_target}' normalizada (MinMaxScaler).")

    print("\n[INFO] Padronizando colunas de volume e quantidade...")
    standard_scaler = StandardScaler()
    df[colunas_padronizar] = standard_scaler.fit_transform(df[colunas_padronizar])
    print(f"[INFO] Colunas padronizadas (StandardScaler): {colunas_padronizar}")

    os.makedirs(path_scalers, exist_ok=True)
    joblib.dump(minmax_scaler, os.path.join(path_scalers, 'minmax_scaler.pkl'))
    joblib.dump(standard_scaler, os.path.join(path_scalers, 'standard_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(path_scalers, 'target_scaler.pkl'))

    joblib.dump({
        'colunas_normalizadas': colunas_normalizar,
        'colunas_padronizadas': colunas_padronizar,
        'colunas_dummy': [col for col in df.columns if col.startswith('Dia_semana_')],
        'scaler_target': 'target_scaler.pkl'
    }, os.path.join(path_scalers, 'scalers_info.pkl'))

    os.makedirs(path_normalizados, exist_ok=True)
    df.to_csv(os.path.join(path_normalizados, 'dados_normalizados.csv'), index=False)

    return df, obs_por_dia

def criar_sequencias(df, obs_por_dia, colunas_temporais, path_sequencias):
    X, y, tempos = [], [], []
    obs_entrada = 2 * obs_por_dia

    df['Data'] = pd.to_datetime(df['Data'])
    dias_unicos = df['Data'].dt.date.unique()

    for i in range(len(dias_unicos) - 2):
        d1, d2, d3 = dias_unicos[i], dias_unicos[i+1], dias_unicos[i+2]

        entrada = pd.concat([
            df[df['Data'].dt.date == d1],
            df[df['Data'].dt.date == d2]
        ])
        saida = df[df['Data'].dt.date == d3]

        # Verifica se os dias têm o número esperado de observações (para evitar sequências inconsistentes)
        if len(entrada) != obs_entrada or len(saida) != obs_por_dia:
            continue

        drop_cols = ['Data', 'Hora', 'Ticker'] + [c for c in colunas_temporais if c not in ['Data', 'Hora']]
        entrada_limpa = entrada.drop(columns=[col for col in drop_cols if col in entrada.columns])

        entrada_limpa = entrada_limpa.select_dtypes(include=[np.number])

        X.append(entrada_limpa.values.astype(np.float32))
        y.append(saida['Fechamento'].values.astype(np.float32))

        # Preencher vetor tempos com as datas dos dois dias de entrada concatenados
        tempos.append([d1, d2])

    X = np.array(X)
    y = np.array(y)
    tempos = np.array(tempos, dtype=object)

    os.makedirs(path_sequencias, exist_ok=True)
    np.save(os.path.join(path_sequencias, 'X.npy'), X)
    np.save(os.path.join(path_sequencias, 'y.npy'), y)
    np.save(os.path.join(path_sequencias, 'colunas_temporais.npy'), tempos)

    print(f"[INFO] Sequências geradas: {X.shape[0]} exemplos.")
    print(f"[INFO] Formato de X: {X.shape}")
    print(f"[INFO] Formato de y: {y.shape}")
    print(f"[INFO] Formato de tempos: {tempos.shape}")

    return X, y, tempos

def dividir_dados(X, y, tempos, path_divisao, proporcao_treino=0.7, proporcao_val=0.15):
    total = len(X)
    if total == 0:
        return (np.array([]),)*9

    n_treino = int(total * proporcao_treino)
    n_val = int(total * proporcao_val)

    X_treino, y_treino, t_treino = X[:n_treino], y[:n_treino], tempos[:n_treino]
    X_val, y_val, t_val = X[n_treino:n_treino+n_val], y[n_treino:n_treino+n_val], tempos[n_treino:n_treino+n_val]
    X_teste, y_teste, t_teste = X[n_treino+n_val:], y[n_treino+n_val:], tempos[n_treino+n_val:]

    os.makedirs(path_divisao, exist_ok=True)
    np.save(os.path.join(path_divisao, 'X_treino.npy'), X_treino)
    np.save(os.path.join(path_divisao, 'y_treino.npy'), y_treino)
    np.save(os.path.join(path_divisao, 'temporais_treino.npy'), t_treino)

    np.save(os.path.join(path_divisao, 'X_val.npy'), X_val)
    np.save(os.path.join(path_divisao, 'y_val.npy'), y_val)
    np.save(os.path.join(path_divisao, 'temporais_val.npy'), t_val)

    np.save(os.path.join(path_divisao, 'X_teste.npy'), X_teste)
    np.save(os.path.join(path_divisao, 'y_teste.npy'), y_teste)
    np.save(os.path.join(path_divisao, 'temporais_teste.npy'), t_teste)

    print(f"[INFO] Divisão: treino={len(X_treino)}, val={len(X_val)}, teste={len(X_teste)}")
    return (X_treino, y_treino, t_treino, X_val, y_val, t_val, X_teste, y_teste, t_teste)

def preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto):
    print(f"\n[INÍCIO] Preparando dados LSTM para {ticker} - Versão: {versao_dados}")

    base_path = os.path.join(path_projeto, f'data')
    path_dados_limpos = os.path.join(base_path, f'cleaned/{ticker}/{intervalo}/dados_limpos_{ticker}_{intervalo}.csv')
    path_dados_preparados = os.path.join(base_path, f'prepared/{ticker}/{intervalo}_{versao_dados}')

    path_normalizados = os.path.join(path_dados_preparados, 'dados_normalizados')
    path_scalers = os.path.join(path_dados_preparados, 'scalers')
    path_sequencias = os.path.join(path_dados_preparados, 'sequencias')
    path_divisao = os.path.join(path_dados_preparados, 'treino_teste_val')

    print("Normalizando e padronizando dados...")
    df, obs_por_dia = normalizar_padronizar_dados(ticker, path_dados_limpos, path_scalers, path_normalizados)

    print("Criando sequências...")
    colunas_temporais = ['Data', 'Hora']
    X, y, tempos = criar_sequencias(df, obs_por_dia, colunas_temporais, path_sequencias)

    if X.shape[0] == 0:
        print("[AVISO] Nenhuma sequência foi criada. Verifique as condições e o volume de dados.")
        return None, None, None

    print("Dividindo os dados...")
    dividir_dados(X, y, tempos, path_divisao)

    print("\n[✔] Pipeline finalizado com sucesso!")
    return X, y, tempos

if __name__ == "__main__":
    preparar_dados_lstm(
        ticker='LREN3',
        intervalo='5m',
        versao_dados='v1',
        path_projeto='/content/pregao_predict'
    )
