
# @title Extração e limpeza de dados

import pandas as pd
import os
import logging
import warnings
from datetime import datetime, timedelta
import yfinance as yf

# Configuração para ignorar warnings de tipo FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reduz o nível de log do yfinance para CRITICAL para evitar poluição no output
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

#--------------------------------------------------------------------------------------#

def limpeza_dados(df):
    print('\nIniciando limpeza de dados...')

    # Ajusta o index datetime para o fuso horário de São Paulo e remove informação de fuso
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/Sao_Paulo").tz_localize(None)

    # Cria colunas auxiliares de hora e data
    df['hora'] = df.index.time
    df.index.name = 'datetime'
    df = df.reset_index()
    df['data'] = pd.to_datetime(df['datetime'].dt.date)  

    # Renomeia as colunas do padrão do yfinance para nomes em português
    mapeamento_colunas = {
        'Close': 'fechamento',
        'High': 'maximo',
        'Low': 'minimo',
        'Open': 'abertura',
        'Volume': 'volume'
    }
    df.rename(columns=mapeamento_colunas, inplace=True)

    # Converte preços para float com 2 casas decimais e volume para inteiro
    for col in ['abertura', 'minimo', 'maximo', 'fechamento']:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce', downcast='integer')

    # Reorganiza as colunas 
    df = df[['data', 'hora', 'abertura', 'minimo', 'maximo', 'fechamento', 'volume', 'datetime']]

    print("\nAmostra após criar a coluna ´data´e ´hora´, reorganizar e renomear as colunas:")
    print(df.head(2))

    # Remove duplicatas e linhas com mais de 50% de valores faltantes
    df = df.drop_duplicates(keep='first')
    linhas_removidas = df[df.isna().sum(axis=1) > (df.shape[1] * 0.5)]
    df = df.dropna(thresh=df.shape[1] * 0.5)
    print(f"Removendo {len(linhas_removidas)} linhas com mais de 50% de NaN")
    if not linhas_removidas.empty:
        print("Datetimes removidos:")
        print(linhas_removidas['datetime'].tolist())

    # Filtra apenas dias úteis (segunda a sexta)
    df = df[df['data'].dt.weekday < 5]
    print("\nAmostra após filtrar apenas dias úteis:")
    print(df.head(4))

    # Filtra horários apenas entre 10:00 e 16:55
    df = df[(df['hora'] >= datetime.strptime('10:05', '%H:%M').time()) &
            (df['hora'] <= datetime.strptime('16:50', '%H:%M').time())]
    print("\nAmostra após filtro de horário (10:05-16:50):")
    print(df.head(4))

    # Validação final
    if df.empty:
        print("O DataFrame ficou vazio após o filtro de horário. Verifique o intervalo de dados.")
    else:
        print("\nLimpeza de dados concluída com sucesso.\n")

    # Ordena por data e hora
    df = df.sort_values(["data", "hora"])
    print("\nAmostra de dados limpos e ordenados:")
    print(df.head(5))
    
    return df

#---------------------- Extração de dados -----------------------------------------#

def extrair_dados(ticker, intervalo, path_projeto, dias_janela=58):
    print('\nExtração de dados...\n')

    # Define caminhos para arquivos brutos e limpos
    path_dados_brutos = os.path.join(path_projeto, f'data/raw/{ticker.split(".")[0]}/dados_brutos_{ticker.split(".")[0]}_{intervalo}.csv')
    path_dados_limpos = os.path.join(path_projeto, f'data/cleaned/{ticker.split(".")[0]}/dados_limpos_{ticker.split(".")[0]}_{intervalo}.csv')

    # Cria um dataframe vazio e define os tempos
    df_total = pd.DataFrame()
    agora = datetime.now()
    hoje = agora.date()
    ontem = hoje - timedelta(days=1)

    # Define data final considerando se o horario do pregão de hoje ja se encerrou
    data_fim = hoje if agora.hour >= 17 and hoje.weekday() < 5 else ontem

    # Define data de início com janela de 58 dias
    data_inicio = datetime.combine(hoje - timedelta(days=dias_janela), datetime.min.time()).replace(hour=10, minute=0)

    primeira_extracao = True
    df_existente = pd.DataFrame()

    # Caso já exista um CSV de dados limpos, atualiza o ponto inicial
    if os.path.exists(path_dados_limpos):
        df_existente = pd.read_csv(path_dados_limpos)
        df_existente["data"] = pd.to_datetime(df_existente["data"], errors="coerce")
        if "data" in df_existente.columns and pd.notnull(df_existente["data"].max()):
            ultimo_registro = df_existente["data"].max().date()
            print(f"Último registro encontrado: {ultimo_registro}")
            data_inicio = ultimo_registro + timedelta(days=1)
            primeira_extracao = False

    print(f"Extraindo dados de {data_inicio} até {data_fim}...")

    # Tenta extrair dados com yfinance
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

    # Corrige index e nomes de colunas do DataFrame extraído
    df_novo.index = pd.to_datetime(df_novo.index, utc=True).tz_convert('America/Sao_Paulo').tz_localize(None)
    df_novo.columns = df_novo.columns.to_flat_index()
    df_novo.columns = [col[0] if isinstance(col, tuple) else col for col in df_novo.columns]

    # Remove a primeira data se for primeira extração (evita dados incompletos)
    if primeira_extracao:
        primeira_data = df_novo.index[0].normalize()
        df_novo = df_novo[df_novo.index.normalize() != primeira_data]
        print("\nAmostra após remoção da primeira data:")
        df_novo = df_novo.sort_index()
        print(df_novo.head(3))

    # Aplica a limpeza de dados
    df_novo_limpo = limpeza_dados(df_novo)

    if primeira_extracao:
        print("\nPrimeira extração de dados.")
        df_total_limpo = df_novo_limpo.copy()
        os.makedirs(os.path.dirname(path_dados_limpos), exist_ok=True)
        os.makedirs(os.path.dirname(path_dados_brutos), exist_ok=True)

        # Salva dados brutos e limpos
        df_total_limpo.to_csv(path_dados_brutos, index=False)
        df_total_limpo.to_csv(path_dados_limpos, index=False)
    else:
        print("\nExtração complementar realizada.")
        df_existente = df_existente[df_novo_limpo.columns]  # Alinha colunas com novo
        df_total_limpo = pd.concat([df_existente, df_novo_limpo])
        df_total_limpo.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
        df_total_limpo = df_total_limpo.sort_values(["data", "hora"])
        df_total_limpo.to_csv(path_dados_limpos, index=False)

    print("\nExtração e limpeza finalizadas com sucesso.")
    return df_total_limpo

# Execução direta para testes
if __name__ == "__main__":
    ticker = 'BBDC4.SA'
    intervalo = '15m'
    path_projeto = "/content/pregao_predict"
    df_total_limpo = extrair_dados(ticker, intervalo, path_projeto)
