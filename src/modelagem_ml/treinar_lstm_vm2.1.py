
#@title Treinar modelo LSTM Versao 2.1 (Normalização dos dados por pregão)
"""
Treinamento do modelo LSTM (versão 2)

Este script realiza o treinamento de um modelo LSTM para prever
os preços de fechamento do próximo pregão, utilizando dados normalizados
e padronizados pregão a pregão.

Funcionalidades principais:
- Carregamento dos dados já processados em sequências por pregão (X, y)
- Carregamento dos scalers específicos por pregão para desnormalização
- Construção do modelo LSTM com duas camadas LSTM, Dropout e Dense final
- Treinamento com EarlyStopping e checkpoint do melhor modelo
- Avaliação e cálculo de métricas (MAE, RMSE, R²) com dados desnormalizados
- Plotagem das previsões versus valores reais para até 6 pregões
- Salvamento do modelo treinado e parâmetros de treinamento
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Gerar horários para o gráfico
def gerar_horarios_pregao_espacado(intervalo_min=5):
    base = datetime(2000, 1, 1, 10, 10)
    return pd.date_range(start=base, end=base.replace(hour=16, minute=45), freq=f'{intervalo_min}min')

# Plotar as predições de alguns pregões
def plotar_predicoes_por_dia(y_true_desnormalizado, y_pred_desnormalizado, datas, horarios_base, titulo='Previsões vs Real', figsize=(32, 7)):
    num_dias = min(6, y_true_desnormalizado.shape[0]) #
    for i in range(num_dias):
        fig, ax = plt.subplots(figsize=figsize)
        horarios_dia = [h + pd.Timedelta(days=i) for h in horarios_base]

        ax.set_ylim(min(y_true_desnormalizado[i]) - 1, max(y_true_desnormalizado[i]) + 1)

        ax.plot(horarios_dia, y_true_desnormalizado[i], label='Real', color='darkgreen', linewidth=3, marker='o', markersize=6)
        ax.plot(horarios_dia, y_pred_desnormalizado[i], label='Previsto', color='darkorange', linestyle='--', linewidth=3, marker='o', markersize=6)

        ax.set_title(f'{titulo} - {datas[i]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Preço', fontsize=12)
        ax.grid(True, which='both', linestyle=':', linewidth=0.8)

        ax.set_xlim([horarios_dia[0], horarios_dia[-1]])
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        ax.legend(fontsize=10, loc='upper left')
        ax.set_xlabel('Horário', fontsize=12)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(titulo, fontsize=16, fontweight='bold')
        plt.show()

def treinar_lstm(ticker, intervalo, versao_modelo, versao_dados, path_projeto, parametros: dict):
    print("\n--- Iniciando treinamento do modelo LSTM ---")
    print(f"[INFO] Versão do modelo: {versao_modelo}")
    print(f"[INFO] Versão dos dados: {versao_dados}")
    print(f"[INFO] Parâmetros de treinamento: {parametros}")

    # Caminhos
    path_modelo_pasta = os.path.join(path_projeto, f'models/LSTM/{ticker}/modelo_lstm_{versao_modelo}')
    path_modelo = os.path.join(path_modelo_pasta, f'modelo_lstm_{versao_modelo}.h5')
    path_parametros_modelo = os.path.join(path_modelo_pasta, f'parametros_{versao_modelo}.pkl')
    path_treino_teste_val = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}/treino_teste_val')
    path_scalers = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}/scalers')

    os.makedirs(path_modelo_pasta, exist_ok=True)
    print(f"[INFO] Pasta do modelo criada/verificada em: {path_modelo_pasta}")

    # Ler dados de treino e validação
    try:
        X_treino = np.load(os.path.join(path_treino_teste_val, 'X_treino.npy'), allow_pickle=True)
        y_treino = np.load(os.path.join(path_treino_teste_val, 'y_treino.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(path_treino_teste_val, 'X_val.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(path_treino_teste_val, 'y_val.npy'), allow_pickle=True)

        # Conversão explícita para tipo float32 (razao?)
        X_treino = np.array(X_treino).astype(np.float32)
        y_treino = np.array(y_treino).astype(np.float32)
        X_val = np.array(X_val).astype(np.float32)
        y_val = np.array(y_val).astype(np.float32)

        print(f"[INFO] Dados carregados e convertidos com sucesso de: {path_treino_teste_val}")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados: {e}")
        return None, None

    try:
        scalers_por_pregao = joblib.load(os.path.join(path_scalers, 'scalers_por_pregao.pkl'))
        print(f"[INFO] Scalers por pregão carregados de: {path_scalers}")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar scalers: {e}")
        return None, None

    num_time_steps = X_treino.shape[1]
    num_features = X_treino.shape[2]
    print(f"[INFO] X_treino.shape = {X_treino.shape}, y_treino.shape = {y_treino.shape}")

    otimizador = Adam(learning_rate=parametros['taxa_aprendizado'])
    funcao_perda = Huber()

    print("[INFO] Construindo arquitetura LSTM...")
    modelo = Sequential([
        LSTM(parametros['unidades_lstm_camada1'], return_sequences=True, input_shape=(num_time_steps, num_features)),
        Dropout(parametros['taxa_dropout']),
        LSTM(parametros['unidades_lstm_camada2'], return_sequences=False),
        Dropout(parametros['taxa_dropout']),
        Dense(64, activation='relu'),
        Dense(y_treino.shape[1])
    ])
    modelo.compile(optimizer=otimizador, loss=funcao_perda)
    modelo.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=parametros['paciencia'], restore_best_weights=True),
        ModelCheckpoint(path_modelo, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("[INFO] Iniciando treinamento...")
    historico = modelo.fit(
        X_treino, y_treino,
        validation_data=(X_val, y_val),
        epochs=parametros['epocas_maximas'],
        batch_size=parametros['tamanho_lote'],
        callbacks=callbacks,
        verbose=1
    )

    y_pred_val = modelo.predict(X_val, verbose=0)

    y_val_desnormalizado, y_pred_desnormalizado = [], []
    for i in range(len(y_val)):
        key = list(scalers_por_pregao.keys())[i + 1]  # target é o próximo pregão
        scaler = scalers_por_pregao[key]['scaler_target_y']
        y_val_desnormalizado.append(scaler.inverse_transform(y_val[i].reshape(-1, 1)).flatten())
        y_pred_desnormalizado.append(scaler.inverse_transform(y_pred_val[i].reshape(-1, 1)).flatten())

    y_val_desnormalizado = np.array(y_val_desnormalizado)
    y_pred_desnormalizado = np.array(y_pred_desnormalizado)

    mae = mean_absolute_error(y_val_desnormalizado, y_pred_desnormalizado)
    rmse = np.sqrt(mean_squared_error(y_val_desnormalizado, y_pred_desnormalizado))
    r2 = r2_score(y_val_desnormalizado, y_pred_desnormalizado)

    print("\n[✓] Métricas de validação:")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R²   = {r2:.6f}")

    print("\n[INFO] Previsões de um único pregão desnormalizadas (10 primeiras observações):")
    for j in range(min(10, y_val_desnormalizado.shape[1])):
        print(f"Obs {j+1:2d} | Real: {y_val_desnormalizado[0, j]:.2f} | Previsto: {y_pred_desnormalizado[0, j]:.2f}")

    horarios = gerar_horarios_pregao_espacado(intervalo_min=5)
    datas = [f"Dia {i+1}" for i in range(y_val.shape[0])]
    plotar_predicoes_por_dia(y_val_desnormalizado, y_pred_desnormalizado, datas, horarios)

    config_modelo = modelo.get_config()
    parametros_modelo = {
        'ticker': ticker,
        'versao': versao_modelo,
        'path_modelo': path_modelo,
        'path_projeto': path_projeto,
        'epocas_executadas': len(historico.history['loss']),
        'loss_final_treino': historico.history['loss'][-1],
        'loss_final_val': historico.history['val_loss'][-1] if 'val_loss' in historico.history else None,
        'layers': [layer['class_name'] for layer in config_modelo['layers']]
    }

    with open(path_parametros_modelo, 'wb') as f:
        pickle.dump(parametros_modelo, f)

    print(f"[✓] Parâmetros de treinamento salvos em: {path_parametros_modelo}")
    return modelo, parametros_modelo

if __name__ == '__main__':
    parametros = {
        "dias_entrada": 1,
        "observacoes_por_dia": 80,
        "num_features_por_observacao": 7,
        "unidades_lstm_camada1": 256,
        "unidades_lstm_camada2": 128,
        "taxa_dropout": 0.01,
        "otimizador": "Adam",
        "taxa_aprendizado": 0.003,
        "funcao_perda": "Huber",
        "paciencia": 15,
        "epocas_maximas": 100,
        "tamanho_lote": 32
    }

    ticker = 'LREN3'
    intervalo = '5m'
    versao_modelo = 'vm2.1'
    versao_dados = 'v2'
    path_projeto = '/content/pregao_predict'

    treinar_lstm(ticker, intervalo, versao_modelo, versao_dados, path_projeto, parametros)
