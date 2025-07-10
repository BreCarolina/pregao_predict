
"""
Avaliar modelo LSTM no conjunto de teste.

Este script:
- Carrega o modelo treinado e os dados de teste (X_teste, y_teste)
- Desnormaliza os dados usando os scalers por pregão
- Calcula métricas de desempenho (MAE, RMSE, R^2)
- Plota as previsões vs reais para até 6 pregões
- Plota os candles reais e previstos no estilo Profit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime


def gerar_horarios_pregao_espacado(intervalo_min=5):
    base = datetime(2000, 1, 1, 10, 10)
    return pd.date_range(start=base, end=base.replace(hour=16, minute=45), freq=f'{intervalo_min}min')


def plotar_predicoes_por_dia(y_true_desnormalizado, y_pred_desnormalizado, datas, horarios_base, titulo='Previsões vs Real', figsize=(32, 7)):
    num_dias = min(6, y_true_desnormalizado.shape[0])
    for i in range(num_dias):
        fig, ax = plt.subplots(figsize=figsize)
        horarios_dia = [h + pd.Timedelta(days=i) for h in horarios_base]

        ax.set_ylim(min(y_true_desnormalizado[i]) - 1, max(y_true_desnormalizado[i]) + 1)

        ax.plot(horarios_dia, y_true_desnormalizado[i], label='Real', color='darkgreen', linewidth=2)
        ax.plot(horarios_dia, y_pred_desnormalizado[i], label='Previsto', color='darkorange', linestyle='--', linewidth=2)

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


def avaliar_lstm_no_teste(ticker, intervalo, versao_modelo, versao_dados, path_projeto):
    print("\n[INICIO] Avaliação do modelo no conjunto de teste")

    path_modelo = os.path.join(path_projeto, f'models/LSTM/{ticker}/modelo_lstm_{versao_modelo}/modelo_lstm_{versao_modelo}.h5')
    path_dados = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}/treino_teste_val')
    path_scalers = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}/scalers')

    modelo = load_model(path_modelo)
    X_teste = np.load(os.path.join(path_dados, 'X_teste.npy'), allow_pickle=True)
    y_teste = np.load(os.path.join(path_dados, 'y_teste.npy'), allow_pickle=True)
    scalers_por_pregao = joblib.load(os.path.join(path_scalers, 'scalers_por_pregao.pkl'))

    y_pred_teste = modelo.predict(X_teste, verbose=0)

    y_teste_desnormalizado, y_pred_desnormalizado = [], []
    for i in range(len(y_teste)):
        key = list(scalers_por_pregao.keys())[i + len(y_teste) * -1]  # Ultimos N pregões são de teste
        scaler = scalers_por_pregao[key]['scaler_target_y']
        y_teste_desnormalizado.append(scaler.inverse_transform(y_teste[i].reshape(-1, 1)).flatten())
        y_pred_desnormalizado.append(scaler.inverse_transform(y_pred_teste[i].reshape(-1, 1)).flatten())

    y_teste_desnormalizado = np.array(y_teste_desnormalizado)
    y_pred_desnormalizado = np.array(y_pred_desnormalizado)

    mae = mean_absolute_error(y_teste_desnormalizado, y_pred_desnormalizado)
    rmse = np.sqrt(mean_squared_error(y_teste_desnormalizado, y_pred_desnormalizado))
    r2 = r2_score(y_teste_desnormalizado, y_pred_desnormalizado)

    print("\n[✓] Métricas no conjunto de TESTE:")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R²   = {r2:.6f}")


    horarios = gerar_horarios_pregao_espacado(intervalo_min=5)
    datas = [f"Teste {i+1}" for i in range(len(y_teste))]
    plotar_predicoes_por_dia(y_teste_desnormalizado, y_pred_desnormalizado, datas, horarios)

if __name__ == '__main__':
    avaliar_lstm_no_teste(
        ticker='LREN3',
        intervalo='5m',
        versao_modelo='vm2.1',
        versao_dados='v2',
        path_projeto='/content/pregao_predict'
    )
