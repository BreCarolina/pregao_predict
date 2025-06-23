
# @title Extração e limpeza de dados

import os
import logging
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Configurações
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

#--------------------------------------------------------------------------------------#

def limpeza_dados(df):
    print('\nIniciando limpeza de dados...')

    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/Sao_Paulo").tz_localize(None)

    df['hora'] = df.index.time
    df.index.name = 'datetime'
    df = df.reset_index()
    df['data'] = pd.to_datetime(df['datetime'].dt.date)

    print("\nAmostra após criação das colunas hora e data:")
    print(df.head(4))

    mapeamento_colunas = {
        'Close': 'fechamento',
        'High': 'maximo',
        'Low': 'minimo',
        'Open': 'abertura',
        'Volume': 'volume'
    }
    df.rename(columns=mapeamento_colunas, inplace=True)

    for col in ['abertura', 'minimo', 'maximo', 'fechamento']:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce', downcast='integer')

    df = df[['data', 'hora', 'abertura', 'minimo', 'maximo', 'fechamento', 'volume', 'datetime']]

    print("\nAmostra após reorganizar as colunas:")
    print(df.head(2))

    df = df.drop_duplicates(keep='first')
    linhas_removidas = df[df.isna().sum(axis=1) > (df.shape[1] * 0.5)]
    df = df.dropna(thresh=df.shape[1] * 0.5)
    print(f"Removendo {len(linhas_removidas)} linhas com mais de 50% de NaN")
    if not linhas_removidas.empty:
        print("Datetimes removidos:")
        print(linhas_removidas['datetime'].tolist())

    print("\nAmostra após remover duplicatas e nulos:")
    print(df.head(2))

    df = df[df['data'].dt.weekday < 5]
    print("\nAmostra após filtrar apenas dias úteis:")
    print(df.head(2))

    df = df[(df['hora'] >= datetime.strptime('10:00', '%H:%M').time()) &
            (df['hora'] <= datetime.strptime('16:55', '%H:%M').time())]
    print("\nAmostra após filtro de horário (10:00-16:55):")
    print(df.head(4))

    if df.empty:
        print("O DataFrame ficou vazio após o filtro de horário. Verifique o intervalo de dados.")
    else:
        print("\nLimpeza de dados concluída com sucesso.\n")

    df = df.sort_values(["data", "hora"])
    print("\nAmostra de dados limpos e ordenados:")
    print(df.head(5))
    print(df.tail(5))

    return df

#--------------------------------------------------------------------------------------#

def extrair_dados(ticker, intervalo, path_projeto, dias_janela=58):
    print('\nExtração de dados...\n')

    path_dados_brutos = os.path.join(path_projeto, f'data/raw/{ticker.split(".")[0]}/dados_brutos_{ticker.split(".")[0]}.csv')
    path_dados_limpos = os.path.join(path_projeto, f'data/cleaned/{ticker.split(".")[0]}/dados_limpos_{ticker.split(".")[0]}.csv')

    df_total = pd.DataFrame()
    agora = datetime.now()
    hoje = agora.date()
    ontem = hoje - timedelta(days=1)

    data_fim = hoje if agora.hour >= 17 and hoje.weekday() < 5 else ontem
    data_inicio = datetime.combine(hoje - timedelta(days=dias_janela), datetime.min.time()).replace(hour=10, minute=0)

    primeira_extracao = True
    df_existente = pd.DataFrame()

    if os.path.exists(path_dados_limpos):
        df_existente = pd.read_csv(path_dados_limpos)
        df_existente["data"] = pd.to_datetime(df_existente["data"], errors="coerce")
        if "data" in df_existente.columns and pd.notnull(df_existente["data"].max()):
            ultimo_registro = df_existente["data"].max().date()
            print(f"Último registro encontrado: {ultimo_registro}")
            data_inicio = ultimo_registro + timedelta(days=1)
            primeira_extracao = False

    print(f"Extraindo dados de {data_inicio} até {data_fim}...")

    try:
        df_novo = yf.download(
            ticker,
            start=data_inicio.strftime("%Y-%m-%d"),
            end=(data_fim + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=intervalo,
            progress=False,
        )
    except Exception as e:
        print(f"Erro ao extrair dados do Yahoo Finance: {e}")
        return df_existente if not df_existente.empty else pd.DataFrame()

    print(f"\nDados extraídos: {len(df_novo)} registros.")

    if df_novo.empty:
        print("\nNenhum dado complementar foi adicionado.")
        return df_existente if not df_existente.empty else pd.DataFrame()

    # Garantir index e colunas corretos
    df_novo.index = pd.to_datetime(df_novo.index, utc=True).tz_convert('America/Sao_Paulo').tz_localize(None)
    df_novo.columns = df_novo.columns.to_flat_index()
    df_novo.columns = [col[0] if isinstance(col, tuple) else col for col in df_novo.columns]

    if primeira_extracao:
        primeira_data = df_novo.index[0].normalize()
        df_novo = df_novo[df_novo.index.normalize() != primeira_data]
        print("\nAmostra após remoção da primeira data:")
        df_novo = df_novo.sort_index()
        print(df_novo.head(3))

    df_novo_limpo = limpeza_dados(df_novo)

    if primeira_extracao:
        print("\nPrimeira extração de dados.")
        df_total_limpo = df_novo_limpo.copy()
        os.makedirs(os.path.dirname(path_dados_limpos), exist_ok=True)
        os.makedirs(os.path.dirname(path_dados_brutos), exist_ok=True)
        df_total_limpo.to_csv(path_dados_brutos, index=False)
        df_total_limpo.to_csv(path_dados_limpos, index=False)
    else:
        print("\nExtração complementar realizada.")
        df_existente = df_existente[df_novo_limpo.columns]  # garantir colunas consistentes
        df_total_limpo = pd.concat([df_existente, df_novo_limpo])
        df_total_limpo.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
        df_total_limpo = df_total_limpo.sort_values(["data", "hora"])
        df_total_limpo.to_csv(path_dados_limpos, index=False)

    print("\nExtração e limpeza finalizadas com sucesso.")
    return df_total_limpo

# Execução direta (testes)
if __name__ == "__main__":
    ticker = 'BBDC4.SA'
    intervalo = '5m'
    path_projeto = "/content/Piloto_day_trade"
    df_total_limpo = extrair_dados(ticker, intervalo, path_projeto)
