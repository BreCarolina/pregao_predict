#@title SRC Transformação de dados para LSTM

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalizar_padronizar_dados(ticker, path_dados_limpos, path_scalers, path_normalizados):
    df = pd.read_csv(path_dados_limpos)    
    df['data'] = pd.to_datetime(df['data'])

    # Verificando a quantidade de observações por dia (Candles)
    obs_por_dia = df.groupby(df['data']).dt.date.size()
    print(f"[INFO] Total de observações por dia:\n{obs_por_dia}")
    
    print("\n[INFO] Criando variáveis categóricas para dias da semana...")
    df['dia_semana'] = df['data'].dt.day_name()
    df = pd.get_dummies(df, columns=['dia_semana'], prefix='dia_semana') 
    print("[INFO] Variáveis categóricas criadas para dias da semana.")
    
    colunas_uteis = ['Data', 'Hora', 'Abertura', 'Máximo', 'Mínimo', 'Volume', 'Quantidade']    
    df = df[colunas_uteis]

    colunas_temporais = ['data', 'hora']
    colunas_normalizar = ['Abertura', 'Máximo', 'Mínimo']
    colunas_padronizar = ['Volume', 'Quantidade']
    coluna_target = 'fechamento'

    for col in colunas_normalizar + colunas_padronizar + [coluna_target]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Coluna '{col}' não é numérica.")

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

    # Salvando os scalers
    os.makedirs(path_scalers, exist_ok=True)
    joblib.dump(minmax_scaler, os.path.join(path_scalers, 'minmax_scaler.pkl'))
    joblib.dump(standard_scaler, os.path.join(path_scalers, 'standard_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(path_scalers, 'target_scaler.pkl'))
    print(f"[INFO] Scalers salvos em: {path_scalers}")

    joblib.dump({
        'colunas_normalizadas': colunas_normalizar,
        'colunas_padronizadas': colunas_padronizar,
        'colunas_dummy': [col for col in df.columns if col.startswith('dia_semana_')],
        'scaler_target': 'target_scaler.pkl'
    }, os.path.join(path_scalers, 'scalers_info.pkl'))
    print(f"[INFO] Informações dos scalers salvas em: {path_scalers}")

    # Salvando uma cópia dos dados normalizados
    os.makedirs(path_normalizados, exist_ok=True)
    df.to_csv(os.path.join(path_normalizados, 'dados_normalizados.csv'), index=False)
    print(f"[INFO] Dados normalizados salvos em: {path_normalizados}")

    return df, int(obs_por_dia) # Retorna o df e a observação por dia calculada

def criar_sequencias(df, obs_por_dia, colunas_temporais, path_sequencias):
    X, y, tempos = [], [], []
    obs_entrada = 2 * obs_por_dia
    obs_saida = obs_por_dia

    dias_unicos = df['data'].dt.date.unique()

    for i in range(len(dias_unicos) - 2):
        # Define dias para entrada (2) e saída (1)
        d1, d2, d3 = dias_unicos[i], dias_unicos[i+1], dias_unicos[i+2]

        # Seleciona dados
        entrada = pd.concat([
            df[df['data'].dt.date == d1],
            df[df['data'].dt.date == d2]
        ])
        saida = df[df['data'].dt.date == d3]

        # Remove colunas temporais da entrada
        drop_cols = ['data', 'hora'] + [c for c in colunas_temporais if c not in ['data', 'hora']]
        X.append(entrada.drop(columns=drop_cols).values.astype(np.float32))
        y.append(saida['Fechamento'].values.astype(np.float32))

    # Converte para arrays
    X = np.array(X)
    y = np.array(y)
    tempos = np.array(tempos, dtype=object)

    # Salva as sequencias
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
        print("[AVISO] Não há sequências para dividir. Retornando DataFrames vazios.")
        return (np.array([]),)*9 # Retorna 9 arrays vazios

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

    # Configurar diretórios
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
    colunas_temporais = ['data', 'hora'] 
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
        ticker='BBDC4',
        intervalo='5m',
        versao_dados='v1',
        path_projeto='/content/pregao_predict'
    )
