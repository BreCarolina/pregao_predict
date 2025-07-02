
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam # Importe Adam diretamente para Huber
from tensorflow.keras.losses import Huber # Importe Huber diretamente

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def treinar_lstm(ticker, intervalo, versao_modelo, versao_dados, path_projeto, parametros: dict):
    print("\n--- Iniciando treinamento do modelo LSTM ---")

    # Caminhos
    path_modelo_pasta = os.path.join(path_projeto, f'models/LSTM/{ticker}/modelo_lstm_{versao_modelo}')
    path_modelo = os.path.join(path_modelo_pasta, f'modelo_lstm_{versao_modelo}.h5')
    path_parametros_modelo = os.path.join(path_modelo_pasta, f'parametros_{versao_modelo}.pkl')
    path_treino_teste_val = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}/treino_teste_val')
    path_scalers = os.path.join(path_projeto, f'data/prepared/{ticker}/{intervalo}_{versao_dados}/scalers')

    os.makedirs(path_modelo_pasta, exist_ok=True)
    print(f"[INFO] Pasta do modelo criada/verificada em: {path_modelo_pasta}")

    # Carregar dados
    try:
        X_treino = np.load(os.path.join(path_treino_teste_val, 'X_treino.npy'))
        y_treino = np.load(os.path.join(path_treino_teste_val, 'y_treino.npy'))
        X_val = np.load(os.path.join(path_treino_teste_val, 'X_val.npy'))
        y_val = np.load(os.path.join(path_treino_teste_val, 'y_val.npy'))
        print(f"[INFO] Dados de treino e validação carregados de: {path_treino_teste_val}")
    except FileNotFoundError as e:
        print(f"[ERRO] Arquivos de dados não encontrados. Verifique o path: {e}")
        print(f"Path procurado: {path_treino_teste_val}")
        return None, None
    except Exception as e:
        print(f"[ERRO] Erro ao carregar dados: {e}")
        return None, None

    # Carregar scaler do fechamento (target)
    try:
        target_scaler = joblib.load(os.path.join(path_scalers, 'target_scaler.pkl'))
        print(f"[INFO] Scaler do target carregado de: {path_scalers}")
    except FileNotFoundError as e:
        print(f"[ERRO] Scaler do target não encontrado. Verifique o path: {e}")
        print(f"Path procurado: {path_scalers}")
        return None, None
    except Exception as e:
        print(f"[ERRO] Erro ao carregar scaler: {e}")
        return None, None

    num_time_steps = X_treino.shape[1] # A segunda dimensão do X_treino é o número de passos de tempo
    num_features = X_treino.shape[2]   # A terceira dimensão do X_treino é o número de features

    # Debug prints para shape antes reshape (agora mais para confirmação)
    print(f"\n[INFO] Shapes dos dados carregados:")
    print(f"X_treino.shape: {X_treino.shape}")
    print(f"y_treino.shape: {y_treino.shape}")
    print(f"X_val.shape: {X_val.shape}")
    print(f"y_val.shape: {y_val.shape}")

    print(f"\n[INFO] Parâmetros inferidos dos dados:")
    print(f"Número de passos de tempo (num_time_steps): {num_time_steps}")
    print(f"Número de features por observação (num_features): {num_features}")
    print(f"Número de observações na saída (y_treino.shape[1]): {y_treino.shape[1]}")


    # Otimizador
    print(f"\n[INFO] Configurando otimizador: {parametros['otimizador']} com taxa de aprendizado {parametros['taxa_aprendizado']}")
    if parametros['otimizador'].lower() == 'adam':
        otimizador = Adam(learning_rate=parametros['taxa_aprendizado'])
    else:
        raise ValueError(f"Otimizador '{parametros['otimizador']}' não implementado.")

    # Função de perda
    print(f"[INFO] Configurando função de perda: {parametros['funcao_perda']}")
    if parametros['funcao_perda'].lower() == 'huber':
        funcao_perda = Huber()
    else:
        raise ValueError(f"Função de perda '{parametros['funcao_perda']}' não implementada.")

    # Construção do modelo
    print("[INFO] Construindo o modelo LSTM...")
    modelo = Sequential([
        LSTM(parametros['unidades_lstm_camada1'], return_sequences=True,
             input_shape=(num_time_steps, num_features)), # Usar os valores inferidos
        Dropout(parametros['taxa_dropout']),
        LSTM(parametros['unidades_lstm_camada2'], return_sequences=False),
        Dropout(parametros['taxa_dropout']),
        Dense(64, activation='relu'),
        Dense(y_treino.shape[1]) # Saída deve ser igual ao número de observações previstas (15)
    ])
    modelo.compile(optimizer=otimizador, loss=funcao_perda)
    modelo.summary() # Adicionar resumo do modelo para depuração

    # Callbacks
    print(f"\n[INFO] Configurando callbacks (EarlyStopping com paciência {parametros['paciencia_parada_antecipada']})...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=parametros['paciencia_parada_antecipada'], restore_best_weights=True),
        ModelCheckpoint(path_modelo, monitor='val_loss', save_best_only=True, verbose=1) # verbose=1 para ver quando salva
    ]

    # Treinamento
    print(f"[INFO] Iniciando treinamento (Épocas Máximas: {parametros['epocas_maximas']}, Tamanho do Lote: {parametros['tamanho_lote']})...")
    historico = modelo.fit(
        X_treino, y_treino,
        validation_data=(X_val, y_val),
        epochs=parametros['epocas_maximas'],
        batch_size=parametros['tamanho_lote'],
        callbacks=callbacks,
        verbose=1 # Mostra o progresso do treinamento
    )

    print(f"\n[✓] Modelo treinado e salvo em: {path_modelo}")
    print(f"Total de épocas executadas: {len(historico.history['loss'])}")
    if 'val_loss' in historico.history and historico.history['val_loss']:
        print(f"Melhor Val Loss atingido: {min(historico.history['val_loss']):.6f}")


    # Avaliação no conjunto de validação
    print("\n[INFO] Realizando previsões no conjunto de validação para avaliação...")
    y_pred_val = modelo.predict(X_val, verbose=0)

    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)

    print("\n[✓] Métricas de desempenho (validação):")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R²   = {r2:.6f}")

    print("\n[INFO] Preparando dados para visualização da primeira amostra...")
    if y_val.shape[0] > 0: # Garante que há pelo menos uma amostra
        y_real_plot = target_scaler.inverse_transform(y_val[0].reshape(-1, 1)).flatten()
        y_pred_plot = target_scaler.inverse_transform(y_pred_val[0].reshape(-1, 1)).flatten()

        plt.figure(figsize=(12, 6)) # Aumentar o tamanho para melhor visualização
        plt.plot(y_real_plot, label='Real (Sequência de Fechamento)', marker='o', linestyle='-')
        plt.plot(y_pred_plot, label='Previsto (Sequência de Fechamento)', marker='x', linestyle='--')
        plt.title('Comparação da Primeira Sequência Prevista vs Real (Validação)')
        plt.xlabel('Observação do Dia (10:15 a 16:50)')
        plt.ylabel('Preço de Fechamento (Escala Original)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("[AVISO] Não há dados de validação suficientes para gerar o gráfico de comparação.")

    # Salvar parâmetros do modelo e métricas
    config_modelo = modelo.get_config()
    parametros_modelo = {
        'ticker': ticker,
        'versao': versao_modelo,
        'path_modelo': path_modelo,
        'path_projeto': path_projeto,
        'epocas_executadas': len(historico.history['loss']),
        'loss_final_treino': historico.history['loss'][-1],
        'loss_final_val': historico.history['val_loss'][-1] if 'val_loss' in historico.history else None,
        'layers': [layer['class_name'] for layer in config_modelo['layers'] if 'class_name' in layer],
        'mae_val': mae,
        'rmse_val': rmse,
        'r2_val': r2,
        'parametros_entrada': parametros # Salvar os parâmetros originais para referência
    }

    with open(path_parametros_modelo, 'wb') as f:
        pickle.dump(parametros_modelo, f)

    print(f"[✓] Parâmetros de treinamento salvos em: {path_parametros_modelo}")
    return modelo, parametros_modelo


# Execução para testes manuais
if __name__ == '__main__':
    parametros = {
        "dias_retrocesso": 2,                       
        "observacoes_por_dia": 84,                  
        "num_features_por_observacao": 9,          
        "unidades_lstm_camada1": 256,
        "unidades_lstm_camada2": 128,
        "taxa_dropout": 0.1,
        "otimizador": "Adam",
        "taxa_aprendizado": 0.001,
        "funcao_perda": "Huber",
        "paciencia_parada_antecipada": 25,
        "epocas_maximas": 300,
        "tamanho_lote": 32
    }
    ticker = 'LREN3'
    intervalo = '5m'
    versao_modelo = 'vm2'
    versao_dados = 'v1'
    path_projeto = '/content/pregao_predict'

    treinar_lstm(ticker, intervalo, versao_modelo, versao_dados, path_projeto, parametros)
