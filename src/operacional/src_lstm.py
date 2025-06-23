
# --- Preparar dados modelo LSTM ---

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Função para normalizar e padronizar os dados transformados
def normalizar_padronizar_dados(ticker, path_dados_transformados, path_scalers, path_dados_normalizados):
    # Garante que o diretório do CSV exista
    os.makedirs(os.path.dirname(path_dados_normalizados), exist_ok=True)

    # Carrega os dados transformados
    df_transformado = pd.read_csv(path_dados_transformados)

    # Colunas relevantes 
    df = df_transformado[['data', 'hora', 'abertura_candle', 'min_candle', 'max_candle',
                          'fechamento_candle', 'volume_candle', 'RSI', 'EMA_20', 'retorno',
                          'hora_num', 'minuto']]

    # Define colunas para normalizar e padronizar
    colunas_normalizar = ['abertura_candle', 'min_candle', 'max_candle', 'fechamento_candle', 'volume_candle']
    colunas_padronizar = ['RSI', 'EMA_20', 'retorno']

    # Instancia os scalers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Aplica os scalers
    df[colunas_normalizar] = minmax_scaler.fit_transform(df[colunas_normalizar])
    df[colunas_padronizar] = standard_scaler.fit_transform(df[colunas_padronizar])

    # Salva scalers
    joblib.dump(minmax_scaler, os.path.join(path_scalers, 'minmax_scaler.pkl'))
    joblib.dump(standard_scaler, os.path.join(path_scalers, 'standard_scaler.pkl'))

    # Salva info sobre colunas transformadas
    scalers_info = {
        'colunas_normalizadas': colunas_normalizar,
        'colunas_padronizadas': colunas_padronizar
    }
    joblib.dump(scalers_info, os.path.join(path_scalers, 'scalers_info.pkl'))

    # Salva os dados normalizados
    df.to_csv(path_dados_normalizados, index=False)
    return df

# Função para criar sequências de 2 pregões para prever o fechamento do pregão seguinte
def criar_sequencias(df, seq_len, target_col, colunas_extra, path_sequencias):
    X, y, extras = [], [], []

    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i + seq_len]
        seq_x = seq.drop(columns=['data', 'hora'] + colunas_extra).values
        seq_y = df.iloc[i + seq_len][target_col]
        X.append(seq_x)
        y.append(seq_y)
        extras.append(df.iloc[i + seq_len][colunas_extra].values)

    # Salva todas as sequências completas (sem divisão)
    np.save(os.path.join(path_sequencias, 'X.npy'), X)
    np.save(os.path.join(path_sequencias, 'y.npy'), y)
    np.save(os.path.join(path_sequencias, 'extras.npy'), extras)

    return np.array(X), np.array(y), np.array(extras)

# Função para dividir os dados em treino, validação e teste
def dividir_dados(X, y, extras, path_treino_teste_val, proporcao_treino=0.7, proporcao_val=0.15):  
    n = len(X)
    n_treino = int(n * proporcao_treino)
    n_val = int(n * proporcao_val)

    X_treino, y_treino, extras_treino = X[:n_treino], y[:n_treino], extras[:n_treino]
    X_val, y_val, extras_val = X[n_treino:n_treino + n_val], y[n_treino:n_treino + n_val], extras[n_treino:n_treino + n_val]
    X_teste, y_teste, extras_teste = X[n_treino + n_val:], y[n_treino + n_val:], extras[n_treino + n_val:]

    # Salva os conjuntos divididos
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

# Função principal que aplica preparação dos dados
def preparar_dados_lstm(ticker, path_projeto):
    # Define caminhos dos arquivos
    path_dados_transformados = os.path.join(path_projeto, f'data/transformed/dados_transformados_{ticker}.csv')
    path_dados_normalizados = os.path.join(path_projeto, f'data/prepared/{ticker}/dados_normalizados.csv')
    path_scalers = os.path.join(path_projeto, f'data/prepared/{ticker}/scalers')
    path_treino_teste_val = os.path.join(path_projeto, f'data/prepared/{ticker}/treino_teste_val')
    path_sequencias = os.path.join(path_projeto, f'data/prepared/{ticker}/sequencias')

    # Cria as pastas, se nao houverem
    os.makedirs(path_scalers, exist_ok=True)
    os.makedirs(path_sequencias, exist_ok=True)
    os.makedirs(path_treino_teste_val, exist_ok=True)

    print("Preparando dados para o modelo LSTM...")

    # Aplica normalização e padronização nos dados transformados
    df = normalizar_padronizar_dados(ticker, path_dados_transformados, path_scalers, path_dados_normalizados)

    # Define o tamanho da sequência, a coluna alvo e extras para salvar
    seq_len = 2
    target_col = 'fechamento_candle'
    colunas_extra = ['data', 'hora']

    # Aplica a função para criar as sequencias
    X, y, extras = criar_sequencias(df, seq_len, target_col, colunas_extra, path_sequencias)

    # Divide as sequencias em treino, validação e teste
    return dividir_dados(X, y, extras, path_treino_teste_val)


#--- Avaliar modelo LSTM ---
import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

def relatorio_acuracia(path_modelo, path_parametros_treinados, path_metricas, path_relatorio_acuracia):
    parametros_treinados = carregar_parametros_treinados(path_parametros_treinados)
    metricas = carregar_metricas(path_metricas)

    if os.path.exists(path_relatorio_acuracia):
        df_acuracia = pd.read_csv(path_relatorio_acuracia)
    else:
        df_acuracia = pd.DataFrame(columns=[
            'modelo_versao', 'mae', 'mse', 'r2',
            'look_back_days', 'expected_obs_per_day', 'num_features_por_observacao_intradia',
            'lstm_units_camada1', 'lstm_units_camada2', 'dropout_rate',
            'optimizer', 'learning_rate', 'loss_function',
            'early_stopping_patience', 'max_epochs', 'batch_size',
            'path_projeto'
        ])

    modelo_versao = os.path.splitext(os.path.basename(path_modelo))[0]

    nova_linha = {
        'modelo_versao': modelo_versao,
        'mae': metricas.get('mae'),
        'mse': metricas.get('mse'),
        'r2': metricas.get('r2'),
        'look_back_days': parametros_treinados.get('look_back_days'),
        'expected_obs_per_day': parametros_treinados.get('expected_obs_per_day'),
        'num_features_por_observacao_intradia': parametros_treinados.get('num_features_por_observacao_intradia'),
        'lstm_units_camada1': parametros_treinados.get('lstm_units_camada1'),
        'lstm_units_camada2': parametros_treinados.get('lstm_units_camada2'),
        'dropout_rate': parametros_treinados.get('dropout_rate'),
        'optimizer': parametros_treinados.get('optimizer'),
        'learning_rate': parametros_treinados.get('learning_rate'),
        'loss_function': parametros_treinados.get('loss_function'),
        'early_stopping_patience': parametros_treinados.get('early_stopping_patience'),
        'max_epochs': parametros_treinados.get('max_epochs'),
        'batch_size': parametros_treinados.get('batch_size'),
        'path_projeto': parametros_treinados.get('path_projeto')
    }

    df_acuracia.loc[len(df_acuracia)] = nova_linha
    df_acuracia.to_csv(path_relatorio_acuracia, index=False)

def carregar_dados_teste(path_treino_teste_val):
    return (
        np.load(os.path.join(path_treino_teste_val, 'X_teste.npy')),
        np.load(os.path.join(path_treino_teste_val, 'y_teste.npy'))
    )

def carregar_modelo(path_modelo):
    return load_model(path_modelo)

def carregar_scaler_precos(path_scalers):
    return joblib.load(path_scalers)

def carregar_parametros_treinados(path_parametros_treinados):
    with open(path_parametros_treinados, 'r') as f:
        return json.load(f)

def carregar_metricas(path_metricas):
    with open(path_metricas, 'r') as f:
        return json.load(f)

def plotar_comparacao_previsoes(
    y_real: np.ndarray,
    y_previsto: np.ndarray,
    obs_por_dia: int,
    titulo_prefixo: str = "",
    max_dias_plot: int = 6
):
    if y_real.shape != y_previsto.shape:
        raise ValueError("Os shapes de y_real e y_previsto devem ser idênticos.")    
    num_dias = min(y_real.shape[0] // obs_por_dia, max_dias_plot)
    if num_dias == 0:
        print("Nenhum dado para plotar.")
        return

    print(f"\n--- Gerando {num_dias} gráficos de comparação (real vs. previsto) ---")
    fig, axes = plt.subplots(nrows=num_dias, ncols=1, figsize=(18, 5 * num_dias))
    if num_dias == 1: 
        axes = [axes] 
    for i in range(num_dias):
        ax = axes[i]
        start = i * obs_por_dia
        end = start + obs_por_dia
        indices = np.arange(obs_por_dia)

        ax.plot(indices, y_real[start:end], label='Real', color='#1f77b4', linewidth=2)
        ax.plot(indices, y_previsto[start:end], label='Previsto', color='#d62728', linestyle='--', linewidth=2)

        ax.set_title(f'Dia {i+1} - {titulo_prefixo} - Previsão de Fechamento (Real vs. Previsto)', fontsize=14)
        ax.set_xlabel('Índice do Candle (5 minutos)', fontsize=12)
        ax.set_ylabel('Fechamento (Desnormalizado)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

def avaliar_modelo_lstm(ticker, versao, path_projeto):
    # Paths necessários
    path_modelo_pasta = os.path.join(path_projeto, f'models/LSTM/{ticker}/modelo_lstm_{versao}')
    path_modelo = os.path.join(path_modelo_pasta, f'modelo_lstm_{versao}.h5')
    path_parametros_treinados = os.path.join(path_modelo_pasta, f'parametros_{versao}.pkl')
    path_metricas = os.path.join(path_modelo_pasta, f'metricas_{versao}.json')
    path_treino_teste_val = os.path.join(path_projeto, f'data/prepared/{ticker}/treino_teste_val')
    path_scalers = os.path.join(path_projeto, f'data/prepared/{ticker}/scalers/minmax_scaler.pkl')
    path_relatorio_acuracia = os.path.join(path_projeto, f'models/LSTM/{ticker}/relatorio_acuracia.csv')

    print("Carregando dados de teste...")
    X_teste, y_teste = carregar_dados_teste(path_treino_teste_val)

    print("Carregando modelo...")
    modelo = carregar_modelo(path_modelo)

    print("Carregando scaler de preços...")
    scaler_precos = carregar_scaler_precos(path_scalers)

    print("Realizando previsões...")
    y_previsto = modelo.predict(X_teste)

    # Inversão da normalização apenas da coluna 'fechamento_candle'
    colunas_precos = ['abertura_candle', 'min_candle', 'max_candle', 'fechamento_candle', 'volume_candle']
    idx_fechamento = colunas_precos.index('fechamento_candle')

    y_previsto_inv = scaler_precos.inverse_transform(
        np.pad(y_previsto.reshape(-1, 1), ((0, 0), (idx_fechamento, len(colunas_precos) - idx_fechamento - 1)), constant_values=0)
    )[:, idx_fechamento]

    y_teste_inv = scaler_precos.inverse_transform(
        np.pad(y_teste.reshape(-1, 1), ((0, 0), (idx_fechamento, len(colunas_precos) - idx_fechamento - 1)), constant_values=0)
    )[:, idx_fechamento]

    df_previsto = pd.DataFrame({'fechamento': y_previsto_inv})
    df_real = pd.DataFrame({'fechamento': y_teste_inv})

    comparacao = pd.DataFrame({
        'Fechamento_Real': df_real['fechamento'],
        'Fechamento_Previsto': df_previsto['fechamento']
    })

    print("\nComparação de previsões (valores reais):")
    print(comparacao.head(10))

    def calcular_metricas(y_real, y_previsto, nome):
        mae = mean_absolute_error(y_real, y_previsto)
        mse = mean_squared_error(y_real, y_previsto)
        r2 = r2_score(y_real, y_previsto)
        print(f"{nome} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        metricas = {'mae': mae, 'mse': mse, 'r2': r2}
        with open(path_metricas, 'w') as f:
            json.dump(metricas, f, indent=4)
        return mae, mse, r2

    print("\nMétricas de desempenho:")
    calcular_metricas(df_real['fechamento'], df_previsto['fechamento'], "Fechamento")

    print("\nGerando relatório de acurácia...")
    relatorio_acuracia(path_modelo, path_parametros_treinados, path_metricas, path_relatorio_acuracia)

    print("\nGerando gráfico de comparação...")
    obs_por_dia = 82
    titulo_prefixo = "LSTM"
    max_dias_plot = 6
    plotar_comparacao_previsoes(df_real['fechamento'].values, df_previsto['fechamento'].values, obs_por_dia, titulo_prefixo, max_dias_plot)

    return df_real, df_previsto, comparacao, carregar_parametros_treinados(path_parametros_treinados)

%%writefile /content/Piloto_day_trade/src/modelagem_ml/treinar_modelo_lstm.py

#@title --- Treinar modelo LSTM ---

import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import warnings

warnings.filterwarnings("ignore")

def treinar_lstm(ticker, versao, path_projeto, parametros_treinamento):
    print("Iniciando treinamento do modelo LSTM...")   
    
    # Paths
    path_modelo_pasta = os.path.join(path_projeto, f'models/LSTM/{ticker}/modelo_lstm_{versao}')    
    path_modelo = os.path.join(path_modelo_pasta, f'modelo_lstm_{versao}.h5')
    path_parametros_modelo = os.path.join(path_modelo_pasta, f'parametros_{versao}.pkl')
    path_treino_teste_val = os.path.join(path_projeto, f'data/prepared/{ticker}/treino_teste_val')

    # Criar pasta do modelo se não existir
    os.makedirs(path_modelo_pasta, exist_ok=True)

    # Carregar dados
    with open(os.path.join(path_treino_teste_val, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(path_treino_teste_val, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(path_treino_teste_val, 'X_val.pkl'), 'rb') as f:
        X_val = pickle.load(f)
    with open(os.path.join(path_treino_teste_val, 'y_val.pkl'), 'rb') as f:
        y_val = pickle.load(f)

    # Redimensionar entradas (esperado: [amostras, steps, features])
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # Inicializar otimizador com learning rate (se aplicável)
    optimizer_name = parametros_treinamento['optimizer'].lower()
    learning_rate = parametros_treinamento['learning_rate']

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Otimizador '{optimizer_name}' não implementado.")

    # Função de perda
    loss_name = parametros_treinamento['loss_function'].lower()
    if loss_name == 'huber':
        loss_function = Huber()
    else:
        raise ValueError(f"Função de perda '{loss_name}' não implementada.")

    # Construir modelo
    modelo = Sequential([
        LSTM(parametros_treinamento['lstm_units_camada1'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(parametros_treinamento['dropout_rate']),
        LSTM(parametros_treinamento['lstm_units_camada2']),
        Dropout(parametros_treinamento['dropout_rate']),
        Dense(y_train.shape[1])
    ])

    modelo.compile(
        optimizer=optimizer,
        loss=loss_function
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=parametros_treinamento['early_stopping_patience'], restore_best_weights=True),
        ModelCheckpoint(path_modelo, monitor='val_loss', save_best_only=True)
    ]

    # Treinamento
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=parametros_treinamento['max_epochs'],
        batch_size=parametros_treinamento['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    print(f"[✓] Modelo treinado e salvo em: {path_modelo}")
    print(f"Total de épocas executadas: {len(history.history['loss'])}")

    # Salvar parâmetros do modelo
    parametros_modelo = parametros_treinamento.copy()
    parametros_modelo['ticker'] = ticker
    parametros_modelo['versao'] = versao
    parametros_modelo['path_modelo'] = path_modelo

    with open(path_parametros_modelo, 'wb') as f:
        pickle.dump(parametros_modelo, f)

    print(f"[✓] Parâmetros de treinamento salvos em: {path_parametros_modelo}")

    return modelo, parametros_modelo


# Execução direta (para testes)
if __name__ == '__main__':
    ticker = 'BBDC4'
    versao = 'v3'
    path_projeto = '/content/Piloto_day_trade'
    parametros_treinamento = {
        "look_back_days": 3,
        "expected_obs_per_day": 82,
        "num_features_por_observacao_intradia": 11,
        "lstm_units_camada1": 64,
        "lstm_units_camada2": 32,
        "dropout_rate": 0.2,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss_function": "Huber",
        "early_stopping_patience": 25,
        "max_epochs": 118,
        "batch_size": 32        
    }

    treinar_lstm(ticker=ticker, versao=versao, path_projeto=path_projeto,
        parametros_treinamento=parametros_treinamento)

