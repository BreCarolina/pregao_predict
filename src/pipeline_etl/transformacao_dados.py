#@title Script de transformação de dados

import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def carregar_dados_limpos(path_dados_limpos: str) -> pd.DataFrame:
    print(f"Carregando dados limpos de: {path_dados_limpos}...")
    try:
        df = pd.read_csv(path_dados_limpos)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')        
        df.dropna(subset=['datetime'], inplace=True)
        df['data'] = pd.to_datetime(df['datetime'].dt.date)
        df['hora'] = df['datetime'].dt.time
        print(f"Dados limpos carregados. Total de {len(df)} registros.")
        return df
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {path_dados_limpos}")
        return pd.DataFrame() 

def carregar_dados_transformados_existentes(path_arquivo_transformado: str, df_limpo: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(path_arquivo_transformado):
        df_transformado_existente = pd.read_csv(path_arquivo_transformado, parse_dates=['datetime', 'data'])        
        df_transformado_existente['data'] = pd.to_datetime(df_transformado_existente['data'])

        ultima_data_transformada = df_transformado_existente['data'].max().date() 
        print(f"[INFO] Última data transformada existente: {ultima_data_transformada}")

        df_novas_datas_a_processar = df_limpo[df_limpo['data'].dt.date > ultima_data_transformada].copy() 
        df_ultima_data_limpo = df_limpo[df_limpo['data'].dt.date == ultima_data_transformada].copy() 
        df_ultima_data_existente = df_transformado_existente[df_transformado_existente['data'].dt.date == ultima_data_transformada].copy() 

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
        return df_raw

def filtrar_janela_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra o DataFrame para incluir apenas registros dentro do intervalo de negociação.
    """
    print("Filtrando registros fora do intervalo 10:05 - 16:50...")
    hora_inicio = pd.to_datetime("10:05").time()
    hora_fim = pd.to_datetime("16:50").time()
    df_filtrado = df[(df['hora'] >= hora_inicio) & (df['hora'] <= hora_fim)].copy()
    print(f"Total de {len(df_filtrado)} registros após filtragem horária.")
    return df_filtrado

def preencher_lacunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche lacunas de tempo (candles ausentes de 5 em 5 minutos) usando a lógica de cópia do vizinho mais próximo.
    """
    print("Preenchendo lacunas de horário...")
    
    horarios_esperados_times = pd.date_range("10:05", "16:50", freq="5min").time

    linhas_adicionadas = []
     
    for data_unica_date_obj in df['data'].dt.date.unique():
        data_unica = data_unica_date_obj 
        df_dia = df[df['data'].dt.date == data_unica].copy()
        
        full_datetimes_esperados = [pd.Timestamp.combine(data_unica, h) for h in horarios_esperados_times]
        
        df_full_day = pd.DataFrame({'datetime': full_datetimes_esperados})
        
        df_dia_merged = pd.merge(df_full_day, df_dia, on='datetime', how='left')

        horarios_faltantes_df = df_dia_merged[df_dia_merged['fechamento_candle'].isna()]

        for index, row in horarios_faltantes_df.iterrows():
            dt_faltante = row['datetime']
            
            anteriores = df_dia[df_dia['datetime'] < dt_faltante]
            posteriores = df_dia[df_dia['datetime'] > dt_faltante]

            copia_base = None

            if not anteriores.empty and not posteriores.empty:
                anterior = anteriores.iloc[-1]
                posterior = posteriores.iloc[0]
                
                if abs((dt_faltante - anterior['datetime']).total_seconds()) <= abs((posterior['datetime'] - dt_faltante).total_seconds()):
                    copia_base = anterior
                else:
                    copia_base = posterior
            elif not anteriores.empty:
                copia_base = anteriores.iloc[-1]
            elif not posteriores.empty:
                copia_base = posteriores.iloc[0]
            else:
                continue

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

def carregar_historico_transformado_necessario(df_novas_datas: pd.DataFrame, path_dados_transformados: str, janela_minima: int = 20) -> pd.DataFrame:
    """
    Carrega um histórico recente de dados transformados para garantir o cálculo correto de indicadores.
    Retorna um DataFrame que inclui o histórico e as novas datas.
    """
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

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os indicadores técnicos RSI e EMA, e trata NaNs resultantes.
    """
    print("Calculando indicadores técnicos (RSI, EMA)...")
    df['fechamento_candle'] = pd.to_numeric(df['fechamento_candle'], errors='coerce')
    
    df['RSI'] = RSIIndicator(close=df['fechamento_candle'], window=14).rsi()
    df['EMA_20'] = EMAIndicator(close=df['fechamento_candle'], window=20).ema_indicator()
    
    df['RSI_indicador_faltante'] = df['RSI'].isna().astype(int)
    df['EMA_20_indicador_faltante'] = df['EMA_20'].isna().astype(int)

    df['RSI'] = df['RSI'].interpolate(method='linear', limit_direction='both').fillna(0)
    df['EMA_20'] = df['EMA_20'].interpolate(method='linear', limit_direction='both').fillna(0)
    
    print("Cálculo de indicadores concluído.")
    return df

def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features baseadas em tempo ao DataFrame.
    """
    print("Adicionando features temporais...")
    df['hora_num'] = df['datetime'].dt.hour
    df['minuto'] = df['datetime'].dt.minute
    df['dia_semana'] = df['datetime'].dt.dayofweek
    print("Features temporais adicionadas.")
    return df

def calcular_retorno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o retorno percentual de cada candle.
    """
    print("Calculando retorno percentual por candle...")
    df['retorno'] = df['fechamento_candle'].pct_change()
    df['retorno'].fillna(0, inplace=True) 
    print("Retorno percentual calculado.")
    return df

def calcular_agregados_diarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os valores de abertura, mínima, máxima, fechamento e volume diários.
    """
    print("Calculando agregados diários...")    
    df['abertura_dia'] = df.groupby('data')['abertura_candle'].transform('first')
    df['minima_dia'] = df.groupby('data')['min_candle'].transform('min')
    df['maxima_dia'] = df.groupby('data')['max_candle'].transform('max')
    df['fechamento_dia'] = df.groupby('data')['fechamento_candle'].transform('last')
    df['volume_dia'] = df.groupby('data')['volume_candle'].transform('sum')
    print("Agregados diários calculados.")
    return df

def criar_coluna_binaria_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma coluna binária indicando a presença de outliers em colunas numéricas.
    """
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


def salvar_ultimos_dois_pregoes(df: pd.DataFrame, ticker: str, path_projeto: str) -> None:
    """
    Salva os dados dos dois últimos pregões em formato CSV para uso no modelo.
    """
    df['data'] = pd.to_datetime(df['data'])
    datas_recentes = sorted(df['data'].dt.date.unique())[-2:]
    df_daily = df[df['data'].dt.date.isin(datas_recentes)].copy()

    pasta_saida = os.path.join(path_projeto, f"data/transformed/{ticker}/daily")
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_saida = os.path.join(pasta_saida, "dados_2_ultimos_pregoes.csv")

    df_daily.to_csv(caminho_saida, index=False)
    print(f"[✓] Últimos 2 pregões salvos em: {caminho_saida}")

def transformar_dados(ticker: str, path_projeto: str) -> None:
    print(f"\n--- Transformação de dados para {ticker} iniciada ---\n")    

    path_dados_limpos = os.path.join(path_projeto, f'data/cleaned/{ticker.split(".")[0]}/dados_limpos_{ticker.split(".")[0]}.csv')
    path_dados_transformados = os.path.join(path_projeto, f'data/transformed/{ticker.split(".")[0]}')  
    path_arquivo_transformado = os.path.join(path_dados_transformados, f'dados_transformados_{ticker.split(".")[0]}.csv')  

    df_limpo = carregar_dados_limpos(path_dados_limpos)
    if df_limpo.empty:
        print("[AVISO] Nenhum dado carregado. Encerrando transformação.")
        return

    df_limpo.rename(columns={
        'abertura': 'abertura_candle',
        'minimo': 'min_candle',
        'maximo': 'max_candle',
        'fechamento': 'fechamento_candle',
        'volume': 'volume_candle'
    }, inplace=True)

    df = carregar_dados_transformados_existentes(path_arquivo_transformado, df_limpo)
    if df.empty:
        return

    df = filtrar_janela_horaria(df)
    df = preencher_lacunas(df)

    if os.path.exists(path_arquivo_transformado):
        df_para_indicadores = carregar_historico_transformado_necessario(df, path_arquivo_transformado)
    else:
        df_para_indicadores = df.copy()

    df_para_indicadores = calcular_indicadores(df_para_indicadores)
    df_para_indicadores = adicionar_features_temporais(df_para_indicadores)
    df_para_indicadores = calcular_retorno(df_para_indicadores)
    df_para_indicadores = calcular_agregados_diarios(df_para_indicadores)
    df_para_indicadores = criar_coluna_binaria_outliers(df_para_indicadores)

    if os.path.exists(path_arquivo_transformado):
        df_final = df_para_indicadores[df_para_indicadores['datetime'] >= df['datetime'].min()].copy()
    else:
        df_final = df_para_indicadores.copy()

    colunas_finais = [
        'data', 'hora',
        'abertura_dia', 'minima_dia', 'maxima_dia', 'fechamento_dia', 'volume_dia',
        'abertura_candle', 'min_candle', 'max_candle', 'fechamento_candle', 'volume_candle',
        'RSI', 'EMA_20', 'retorno',
        'dia_semana', 'hora_num', 'minuto', 'datetime', 'is_outlier',
        'RSI_indicador_faltante', 'EMA_20_indicador_faltante'
    ]

    for col in colunas_finais:
        if col not in df_final.columns:
            df_final[col] = np.nan

    df_final = df_final[colunas_finais].copy()
    df_final.fillna(0, inplace=True)

    if os.path.exists(path_arquivo_transformado):
        df_existente_para_anexar = pd.read_csv(path_arquivo_transformado, parse_dates=['datetime', 'data'])        
        df_existente_para_anexar['data'] = pd.to_datetime(df_existente_para_anexar['data'])
        df_salvar = pd.concat([df_existente_para_anexar, df_final], ignore_index=True)
        df_salvar.drop_duplicates(subset=['datetime'], inplace=True)
        df_salvar.sort_values('datetime', inplace=True)
        df_salvar.reset_index(drop=True, inplace=True)
    else:
        df_salvar = df_final.copy()

    os.makedirs(os.path.dirname(path_arquivo_transformado), exist_ok=True)
    df_salvar.to_csv(path_arquivo_transformado,  index=False)
    print(f"\n[✓] Transformação concluída. Dados salvos em: {path_arquivo_transformado}. Total de {len(df_salvar)} registros.")

    salvar_ultimos_dois_pregoes(df_salvar, ticker, path_projeto)

if __name__ == "__main__":
    ticker = 'OIBR3.SA'
    path_projeto = "/content/Piloto_day_trade"  
    transformar_dados(ticker, path_projeto)

