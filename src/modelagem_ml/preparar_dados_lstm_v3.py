#@title Preparar dados LSTM - Normalização, criação de sequencias e divisão dos dados Versao 03

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Função para normalizar e padronizar os dados limpos
def normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados):
    df = pd.read_csv(path_dados_limpos)
    df['Data'] = pd.to_datetime(df['Data'])

    # Filtra os dados para manter apenas registros entre 10:10 e 16:45
    df = df[(df['Hora'] >= '10:10:00') & (df['Hora'] <= '16:45:00')]

    # Conta a quantidade de candles por dia
    contagem_por_dia = df.groupby(df['Data'].dt.date).size()
    frequencia = contagem_por_dia.value_counts()
    print("\n[INFO] Verificando quantidade de observações por dia...")
    print("\n[INFO] Frequência das quantidades de candles por dia:")
    for quantidade, dias in frequencia.items():
        print(f"- {quantidade} candles: {dias} dias")

    # Mantém apenas os dias com 80 candles
    dias_validos = contagem_por_dia[contagem_por_dia == 80].index
    df = df[df['Data'].dt.date.isin(dias_validos)]
    print(f"\n[INFO] Quantidade de candles mais comum: 80 >>>>>>>>>> Vamos filtrar o dataset antes de preparar os dados, e vamos trabalhar somente com 80 candles por dia em todos.")

    # Cria variáveis categóricas para o dia da semana
    print("\n[INFO] Criando variaveis categóricas para dias da semana...")
    df['Dia_semana'] = df['Data'].dt.day_name()
    df = pd.get_dummies(df, columns=['Dia_semana'], prefix='Dia_semana')
    print("[INFO] Variaveis categóricas criadas.")

    # Define colunas para normalizar, padronizar e a coluna target
    colunas_temporais = ['Data', 'Hora'] + [col for col in df.columns if col.startswith('Dia_semana_')]
    colunas_normalizar = ['Abertura', 'Máximo', 'Mínimo']
    colunas_padronizar = ['Volume', 'Quantidade']
    coluna_target = 'Fechamento'

    # Normaliza colunas de preço
    print("\n[INFO] Normalizando colunas de preco...")
    minmax_scaler = MinMaxScaler()
    df[colunas_normalizar] = minmax_scaler.fit_transform(df[colunas_normalizar])

    # Normaliza a coluna target separadamente
    print("\n[INFO] Normalizando coluna target separadamente...")
    target_scaler = MinMaxScaler()
    df[[coluna_target]] = target_scaler.fit_transform(df[[coluna_target]])

    # Padroniza volume e quantidade
    print("\n[INFO] Padronizando colunas de volume e quantidade...")
    standard_scaler = StandardScaler()
    df[colunas_padronizar] = standard_scaler.fit_transform(df[colunas_padronizar])

    # Salva os scalers utilizados
    os.makedirs(path_scalers, exist_ok=True)
    joblib.dump(minmax_scaler, os.path.join(path_scalers, 'minmax_scaler.pkl'))
    joblib.dump(standard_scaler, os.path.join(path_scalers, 'standard_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(path_scalers, 'target_scaler.pkl'))

    # Salva metadados de normalização
    joblib.dump({
        'colunas_normalizadas': colunas_normalizar,
        'colunas_padronizadas': colunas_padronizar,
        'colunas_dummy': [col for col in df.columns if col.startswith('Dia_semana_')],
        'scaler_target': 'target_scaler.pkl'
    }, os.path.join(path_scalers, 'scalers_info.pkl'))

    # Salva os dados tratados
    os.makedirs(path_normalizados, exist_ok=True)
    df.to_csv(os.path.join(path_normalizados, 'dados_normalizados.csv'), index=False)

    return df, colunas_temporais

# Função para criar as sequências LSTM (um pregão para o seguinte)
def criar_sequencias(df, path_sequencias):
    print("[INFO] Criando sequencias com janela por pregao...")
    X, y, tempos = [], [], []

    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by=['Data', 'Hora']).reset_index(drop=True)
    print("[INFO] Dados ordenados.")

    dias_unicos = df['Data'].dt.date.unique()
    print(f"[INFO] Total de dias: {len(dias_unicos)}")
    print("[INFO] Amostra dados ordenados")
    print(df.head())

    colunas_features = [col for col in df.columns if col not in ['Data', 'Hora', 'Ativo']]
    grouped = df.groupby(df['Data'].dt.date)

    for i in range(len(dias_unicos) - 1):
        dia1 = dias_unicos[i]

        # Verifica se o dia atual é sexta-feira e ajusta índice para segunda-feira seguinte
        if pd.to_datetime(dia1).weekday() == 4 and i + 2 < len(dias_unicos):
            dia2 = dias_unicos[i + 2]
        else:
            dia2 = dias_unicos[i + 1]

        df1 = grouped.get_group(dia1)[colunas_features]
        df2 = grouped.get_group(dia2)['Fechamento']

        # Decide qual dos dois dias tem menos candles
        n = min(len(df1), len(df2))

        if n < 10:
            continue

        if len(df1) <= len(df2):
            entrada = df1.iloc[:n]
            saida = df2.iloc[:n]
        else:
            entrada = df1.iloc[:n]
            saida = df2.iloc[:n]

        if entrada.isnull().values.any() or saida.isnull().values.any():
            continue

        X.append(entrada.values.astype(np.float32))
        y.append(saida.values.astype(np.float32))
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

# Função para dividir os dados em treino, validação e teste
def dividir_dados(X, y, tempos, path_divisao, proporcao_treino=0.7, proporcao_val=0.15):
    total = len(X)
    if total == 0:
        print("[AVISO] Nenhum dado para dividir.")
        return (np.array([]),) * 9

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

# Pipeline completo para preparar os dados LSTM
def preparar_dados_lstm(ticker, intervalo, versao_dados, path_projeto):
    print(f"\n[INICIO] Preparando dados LSTM para {ticker} - Versao: {versao_dados}")

    base_path = os.path.join(path_projeto, 'data')
    path_dados_limpos = os.path.join(base_path, f'cleaned/{ticker}/{intervalo}/dados_limpos_{ticker}_{intervalo}.csv')
    path_dados_preparados = os.path.join(base_path, f'prepared/{ticker}/{intervalo}_{versao_dados}')

    path_normalizados = os.path.join(path_dados_preparados, 'dados_normalizados')
    path_scalers = os.path.join(path_dados_preparados, 'scalers')
    path_sequencias = os.path.join(path_dados_preparados, 'sequencias')
    path_divisao = os.path.join(path_dados_preparados, 'treino_teste_val')

    print("Normalizando e padronizando dados...")
    df, colunas_temporais = normalizar_padronizar_dados(path_dados_limpos, path_scalers, path_normalizados)

    print("\nCriando sequencias...")
    X, y, tempos = criar_sequencias(df, path_sequencias)

    if X.shape[0] == 0:
        print("[AVISO] Nenhuma sequencia foi criada. Verifique os dados.")
        return None, None, None

    print("Dividindo os dados...")
    dividir_dados(X, y, tempos, path_divisao)

    print("\n[✔] Pipeline finalizado com sucesso!")
    return X, y, tempos

# Execução direta
def main():
    preparar_dados_lstm(
        ticker='LREN3',
        intervalo='5m',
        versao_dados='v3',
        path_projeto='/content/pregao_predict'
    )

if __name__ == "__main__":
    main()
