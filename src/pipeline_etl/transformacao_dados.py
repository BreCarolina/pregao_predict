
#@title Script de transformação de dados

import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import warnings

# Suprime alertas do tipo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Função para carregar os dados limpos 
def carregar_dados_limpos(path_dados_limpos: str) -> pd.DataFrame:
    print(f"Carregando dados limpos de: {path_dados_limpos}...")
    try:
        df = pd.read_csv(path_dados_limpos)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')  # Garante datetime válido
        df.dropna(subset=['datetime'], inplace=True)  # Remove linhas com datetime inválido
        df['data'] = pd.to_datetime(df['datetime'].dt.date)
        df['hora'] = df['datetime'].dt.time
        print(f"Dados limpos carregados. Total de {len(df)} registros.")
        return df
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {path_dados_limpos}")
        return pd.DataFrame()

# Carrega apenas os dados que ainda não foram transformados
def carregar_dados_transformados_existentes(path_arquivo_transformado: str, df_limpo: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(path_arquivo_transformado):
        df_transformado_existente = pd.read_csv(path_arquivo_transformado, parse_dates=['datetime', 'data'])
        df_transformado_existente['data'] = pd.to_datetime(df_transformado_existente['data'])

        ultima_data_transformada = df_transformado_existente['data'].max().date()
        print(f"[INFO] Última data transformada existente: {ultima_data_transformada}")

        # Separa novos dias e novos candles da última data
        df_novas_datas_a_processar = df_limpo[df_limpo['data'].dt.date > ultima_data_transformada].copy()
        df_ultima_data_limpo = df_limpo[df_limpo['data'].dt.date == ultima_data_transformada].copy()
        df_ultima_data_existente = df_transformado_existente[df_transformado_existente['data'].dt.date == ultima_data_transformada].copy()

        # Detecta novos candles da mesma data que ainda não foram processados
        novos_candles_ultima_data = df_ultima_data_limpo[
            ~df_ultima_data_limpo['datetime'].isin(df_ultima_data_existente['datetime'])
        ].copy()

        df = pd.concat([df_novas_datas_a_processar, novos_candles_ultima_data], ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)

        if df.empty:
            print("[AVISO] Nenhuma nova data ou novo candle disponível para transformação. Pipeline encerrado.")
            return pd.DataFrame()

        print(f"[INFO] Novas datas/candles a serem processadas: {df['data'].min().date()} a {df['data'].max().date()}. Total de {len(df)} registros.")
        return df
    else:
        print("[INFO] Nenhum arquivo transformado anterior encontrado. Serão processadas todas as datas disponíveis.")
        return df_limpo.copy()

# Preenche candles ausentes com candle vizinho proximo
def preencher_lacunas(df: pd.DataFrame, intervalo) -> pd.DataFrame:
    print("Preenchendo lacunas de horário...")
    horarios_esperados_times = pd.date_range("10:05", "16:50", freq=intervalo).time
    linhas_adicionadas = []

    for data_unica_date_obj in df['data'].dt.date.unique():
        data_unica = data_unica_date_obj
        df_dia = df[df['data'].dt.date == data_unica].copy()

        full_datetimes_esperados = [pd.Timestamp.combine(data_unica, h) for h in horarios_esperados_times]
        df_full_day = pd.DataFrame({'datetime': full_datetimes_esperados})
        df_dia_merged = pd.merge(df_full_day, df_dia, on='datetime', how='left')
        horarios_faltantes_df = df_dia_merged[df_dia_merged['fechamento'].isna()]

        # Preenche as lacunas usando o candle mais próximo no tempo
        for _, row in horarios_faltantes_df.iterrows():
            dt_faltante = row['datetime']
            anteriores = df_dia[df_dia['datetime'] < dt_faltante]
            posteriores = df_dia[df_dia['datetime'] > dt_faltante]
            copia_base = None

            if not anteriores.empty and not posteriores.empty:
                anterior = anteriores.iloc[-1]
                posterior = posteriores.iloc[0]
                copia_base = anterior if abs((dt_faltante - anterior['datetime']).total_seconds()) <= abs((posterior['datetime'] - dt_faltante).total_seconds()) else posterior
            elif not anteriores.empty:
                copia_base = anteriores.iloc[-1]
            elif not posteriores.empty:
                copia_base = posteriores.iloc[0]

            if copia_base is not None:
                nova_linha = copia_base.copy()
                nova_linha['datetime'] = dt_faltante
                nova_linha['data'] = pd.to_datetime(dt_faltante.date())
                nova_linha['hora'] = dt_faltante.time()
                nova_linha['volume_candle'] = 0
                nova_linha['ajuste'] = 0
                linhas_adicionadas.append(nova_linha)

    if linhas_adicionadas:
        df_adicional = pd.DataFrame(linhas_adicionadas)
        df_adicional['data'] = pd.to_datetime(df_adicional['data'])
        df = pd.concat([df, df_adicional], ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"Adicionadas {len(linhas_adicionadas)} lacunas. Total de {len(df)} registros.")
    else:
        print("Nenhuma lacuna encontrada ou preenchida.")
    return df

# Carrega dados transformados anteriores para cálculo de indicadores com janela (ex: EMA 20)
def carregar_historico_transformado_necessario(df_novas_datas: pd.DataFrame, path_dados_transformados: str, janela_minima: int = 20) -> pd.DataFrame:
    print("[INFO] Carregando histórico transformado para cálculo dos indicadores...")
    try:
        df_historico_transformado = pd.read_csv(path_dados_transformados, parse_dates=['datetime', 'data'])
    except FileNotFoundError:
        print("[AVISO] Arquivo de histórico transformado não encontrado. Prosseguindo apenas com as novas datas.")
        return df_novas_datas

    min_datetime_novas_datas = df_novas_datas['datetime'].min()
    dados_anteriores = df_historico_transformado[df_historico_transformado['datetime'] < min_datetime_novas_datas].copy()
    dados_suficientes = dados_anteriores.tail(janela_minima + 50)
    df_combinado = pd.concat([dados_suficientes, df_novas_datas], ignore_index=True).sort_values('datetime').reset_index(drop=True)

    print(f"Carregados {len(dados_suficientes)} registros históricos para suporte de cálculo.")
    return df_combinado


# Adiciona colunas com informações temporais
def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    print("Adicionando features temporais...")
    df['hora_num'] = df['datetime'].dt.hour
    df['minuto'] = df['datetime'].dt.minute
    df['dia_semana'] = df['datetime'].dt.dayofweek
    print("Features temporais adicionadas.")
    return df

# Calcula o retorno percentual por candle
def calcular_retorno(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculando retorno percentual por candle...")
    df['retorno'] = df['fechamento'].pct_change()
    df['retorno'].fillna(0, inplace=True)
    print("Retorno percentual calculado.")
    return df

# Gera agregados diários: OHLC e volume
def calcular_agregados_diarios(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculando agregados diários...")
    df['abertura_dia'] = df.groupby('data')['abertura'].transform('first')
    df['minima_dia'] = df.groupby('data')['minimo'].transform('min')
    df['maxima_dia'] = df.groupby('data')['maximo'].transform('max')
    df['fechamento_dia'] = df.groupby('data')['fechamento'].transform('last')
    df['volume_dia'] = df.groupby('data')['volume'].transform('sum')
    print("Agregados diários calculados.")
    return df

# Detecta outliers numéricos com base no IQR
def criar_coluna_binaria_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("Criando coluna binária para outliers...")
    exclude_cols = ['is_outlier', 'RSI_indicador_faltante', 'EMA_20_indicador_faltante', 'hora_num', 'minuto', 'dia_semana']
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    colunas_para_outlier = [col for col in colunas_numericas if col not in exclude_cols]
    outlier_mask = pd.Series(False, index=df.index)

    for col in colunas_para_outlier:
        if df[col].isnull().sum() > len(df) * 0.5:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers_col = (df[col] < limite_inferior) | (df[col] > limite_superior)
        outlier_mask = outlier_mask | outliers_col

    df['is_outlier'] = outlier_mask.astype(int)
    print("Coluna 'is_outlier' criada.")
    return df

# Salva os últimos n pregões em CSV (usado para predição futura)
def salvar_ultimos_pregoes(df: pd.DataFrame, ticker: str, path_daily: str) -> None:
    df['data'] = pd.to_datetime(df['data'])
    datas_recentes = sorted(df['data'].dt.date.unique())[-2:]
    df_daily = df[df['data'].dt.date.isin(datas_recentes)].copy()
   
    caminho_saida = os.path.join(path_daily, f"dois_ultimos_pregoes.csv")

    df_daily.to_csv(caminho_saida, index=False)
    print(f"[✓] Daily salvo em: {caminho_saida}")

# Pipeline principal de transformação de dados
def transformar_dados(ticker: str, intervalo: str, path_projeto: str) -> None:
    print(f"\n--- Transformação de dados  iniciada ---\n")

    # Caminhos dos arquivos
    path_dados_limpos = os.path.join(path_projeto, f'data/cleaned/{ticker.split(".")[0]}/dados_limpos_{ticker.split(".")[0]}_{intervalo}.csv')
    path_dados_transformados = os.path.join(path_projeto, f'data/transformed/{ticker.split(".")[0]}/{intervalo}')
    path_arquivo_transformado = os.path.join(path_dados_transformados, f'dados_transformados_{ticker.split(".")[0]}_{intervalo}.csv')
    path_daily = os.path.join(path_dados_transformados,f'daily_{intervalo}')

    os.makedirs(path_daily, exist_ok=True)

    df_limpo = carregar_dados_limpos(path_dados_limpos)
    if df_limpo.empty:
        print("[AVISO] Nenhum dado carregado. Encerrando transformação.")
        return

    df = carregar_dados_transformados_existentes(path_arquivo_transformado, df_limpo)
    if df.empty:
        return

    # Executa as etapas do pipeline
    df = preencher_lacunas(df, intervalo)
    df_para_indicadores = carregar_historico_transformado_necessario(df, path_arquivo_transformado) if os.path.exists(path_arquivo_transformado) else df.copy()
    df_para_indicadores = adicionar_features_temporais(df_para_indicadores)
    df_para_indicadores = calcular_retorno(df_para_indicadores)
    df_para_indicadores = calcular_agregados_diarios(df_para_indicadores)
    df_para_indicadores = criar_coluna_binaria_outliers(df_para_indicadores)

    # Filtra para manter apenas o novo intervalo transformado
    df_final = df_para_indicadores[df_para_indicadores['datetime'] >= df['datetime'].min()] if os.path.exists(path_arquivo_transformado) else df_para_indicadores.copy()

    # Define as colunas finais padronizadas
    colunas_finais = [
        'data', 'hora',
        'abertura_dia', 'minima_dia', 'maxima_dia', 'fechamento_dia', 'volume_dia',
        'abertura', 'minima', 'maxima', 'fechamento', 'volume',
        'retorno', 'dia_semana', 'hora_num', 'minuto', 'datetime', 'is_outlier'        
    ]

    for col in colunas_finais:
        if col not in df_final.columns:
            df_final[col] = np.nan
    df_final = df_final[colunas_finais].copy()
    df_final.fillna(0, inplace=True)

    # Junta com transformado anterior (se existir), remove duplicatas e ordena
    if os.path.exists(path_arquivo_transformado):
        df_existente_para_anexar = pd.read_csv(path_arquivo_transformado, parse_dates=['datetime', 'data'])
        df_existente_para_anexar['data'] = pd.to_datetime(df_existente_para_anexar['data'])
        df_salvar = pd.concat([df_existente_para_anexar, df_final], ignore_index=True)
        df_salvar.drop_duplicates(subset=['datetime'], inplace=True)
        df_salvar.sort_values('datetime', inplace=True)
        df_salvar.reset_index(drop=True, inplace=True)
    else:
        df_salvar = df_final.copy()

    # Salva o arquivo transformado completo
    os.makedirs(os.path.dirname(path_arquivo_transformado), exist_ok=True)
    df_salvar.to_csv(path_arquivo_transformado,  index=False)
    print(f"\n[✓] Transformação concluída. Dados salvos em: {path_arquivo_transformado}. Total de {len(df_salvar)} registros.")

    # Salva últimos n pregões para uso no modelo
    salvar_ultimos_pregoes(df_salvar, ticker, path_projeto)

# Execução direta para testes
if __name__ == "__main__":
    ticker = 'BBDC4'
    intervalo = '15m'
    path_projeto = "/content/pregao_predict"
    transformar_dados(ticker, intervalo, path_projeto)
