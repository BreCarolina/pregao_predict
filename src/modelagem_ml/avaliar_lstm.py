
#@title Script para Avaliar LSTM e abastecer relatório de acurácia

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model


def salvar_arquivo_json(dados, caminho): 
    """
    Salva um dicionário em formato JSON no caminho especificado.
    """
    with open(caminho, 'w') as f:
        json.dump(dados, f, indent=4)


def plotar_comparacao_previsoes(y_true, y_pred, ticker, versao_modelo, path_modelo):
    """
    Plota gráfico de comparação entre valores reais e previstos.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Real')
    plt.plot(y_pred, label='Previsto')
    plt.title(f'Previsão vs Real - {ticker} ({versao_modelo})')
    plt.xlabel('Amostra')
    plt.ylabel('Fechamento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path_modelo, 'grafico_comparacao.png'))
    plt.close()


def desnormalizar_fechamento(y, scaler, idx_feature=3, total_features=5):
    """
    Desnormaliza apenas a feature 'fechamento', assumindo que ela está no índice 3.
    """
    padded = np.pad(y.reshape(-1, 1),
                    ((0, 0), (idx_feature, total_features - idx_feature - 1)),
                    mode='constant')
    return scaler.inverse_transform(padded)[:, idx_feature].reshape(y.shape)


def avaliar_lstm(ticker, versao_modelo, versao_dados, path_projeto):

    path_modelo = os.path.join(path_projeto, f'models/LSTM/{ticker}/modelo_lstm_{versao_modelo}')
    base_dados = os.path.join(path_projeto, f'data/prepared/{ticker}/{versao_dados}')
    path_dados_normalizados = os.path.join(base_dados, 'dados_normalizados')
    path_scalers = os.path.join(base_dados, 'scalers')
    path_sequencias = os.path.join(base_dados, 'sequencias')
    path_treino_teste_val = os.path.join(base_dados, 'treino_teste_val')

    print('[✓] Carregando dados de teste...')
    X_teste, y_teste = joblib.load(path_treino_teste_val)
    X_teste = X_teste.reshape((X_teste.shape[0], X_teste.shape[1], 1))

    print('[✓] Carregando modelo...')
    model_file = os.path.join(path_modelo, f'modelo_lstm_{versao_modelo}.h5')
    model = load_model(model_file)

    print('[✓] Carregando scaler de preços...')
    scaler_file = os.path.join(path_scalers, 'minmax_scaler.pkl')
    scaler = joblib.load(scaler_file)

    print('[✓] Realizando previsões...')
    if len(X_teste.shape) == 2:
        X_teste = X_teste.reshape((X_teste.shape[0], X_teste.shape[1], 1))
    y_previsto = model.predict(X_teste)

    print('[✓] Desnormalizando previsões...')
    y_teste_inv = desnormalizar_fechamento(y_teste, scaler)
    y_previsto_inv = desnormalizar_fechamento(y_previsto, scaler)

    print('[✓] Calculando métricas...')
    y_true_flat = y_teste_inv.flatten()
    y_pred_flat = y_previsto_inv.flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    print(f"Fechamento - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

    print('[✓] Gerando relatório de acurácia...')
    parametros_file = f'parametros_{versao_modelo}.pkl'
    parametros_finais = joblib.load(os.path.join(path_modelo, parametros_file))

    relatorio_completo = {
        'ticker': ticker,
        'versao_modelo': versao_modelo,
        'versao_dados': versao_dados,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'epocas_executadas': parametros_finais.get('epocas_executadas'),
        'loss_final_treino': parametros_finais.get('loss_final_treino'),
        'loss_final_val': parametros_finais.get('loss_final_val'),
        'layers': parametros_finais.get('layers'),
        'path_modelo': parametros_finais.get('path_modelo')
    }

    salvar_arquivo_json(relatorio_completo, os.path.join(path_modelo, 'acuracia.json'))

    print('[✓] Gerando gráfico de comparação...')
    plotar_comparacao_previsoes(y_teste_inv, y_previsto_inv, ticker, versao_modelo, path_modelo)

    print('[✓] Avaliação concluída.')


if __name__ == '__main__':
    ticker = 'BBDC4'
    versao_modelo = 'v1'
    versao_dados = 'v001'
    path_projeto = '/content/pregao_predict'

    avaliar_lstm(ticker, versao_modelo, versao_dados, path_projeto)
