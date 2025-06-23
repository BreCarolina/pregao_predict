import sqlite3
import pandas as pd
from datetime import datetime
import os

def conectar_db(caminho_db):
    """Conecta ou cria o banco de dados SQLite."""
    return sqlite3.connect(caminho_db)

def criar_tabelas(conn):
    """Cria as tabelas no banco de dados se elas não existirem."""
    cursor = conn.cursor()

    # Tabela de dimensão de tempo
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_tempo (
        id_tempo INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT NOT NULL,
        hora TEXT NOT NULL,
        datetime TEXT NOT NULL UNIQUE, -- datetime é a chave de unicidade para o ponto no tempo
        dia_da_semana INTEGER NOT NULL
    );
    """)

    # Tabela de dimensão de indicadores técnicos
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_indicadores (
        id_indicadores INTEGER PRIMARY KEY AUTOINCREMENT,
        RSI REAL NOT NULL,
        EMA_20 REAL NOT NULL,
        retorno REAL NOT NULL,
        UNIQUE(RSI, EMA_20, retorno) -- Garante unicidade dos valores dos indicadores
    );
    """)

    # Tabela de dimensão de agregados diários
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_agregados (
        id_agregados INTEGER PRIMARY KEY AUTOINCREMENT,
        hora_num INTEGER NOT NULL,
        minuto INTEGER NOT NULL,
        abertura_dia REAL NOT NULL,
        minima_dia REAL NOT NULL,
        maxima_dia REAL NOT NULL,
        fechamento_dia REAL NOT NULL,
        volume_dia INTEGER NOT NULL,
        UNIQUE(hora_num, minuto, abertura_dia, minima_dia, maxima_dia, fechamento_dia, volume_dia)
    );
    """)

    # Tabela de fatos para os preços e suas relações com as dimensões
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fato_precos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        abertura REAL NOT NULL,
        minima REAL NOT NULL,
        maxima REAL NOT NULL,
        fechamento REAL REAL NOT NULL,
        volume INTEGER NOT NULL,
        id_tempo INTEGER NOT NULL,
        id_indicadores INTEGER NOT NULL,
        id_agregados INTEGER NOT NULL,
        FOREIGN KEY (id_tempo) REFERENCES dim_tempo(id_tempo),
        FOREIGN KEY (id_indicadores) REFERENCES dim_indicadores(id_indicadores),
        FOREIGN KEY (id_agregados) REFERENCES dim_agregados(id_agregados),
        UNIQUE(id_tempo, id_indicadores, id_agregados) -- Garante unicidade da linha de fato
    );
    """)

    conn.commit()

def obter_id(conn, tabela, valores):
    """
    Obtém o ID de um registro existente na dimensão ou insere um novo, retornando o ID.
    Utiliza INSERT OR IGNORE para otimizar o processo e garantir a unicidade dos dados.
    """
    placeholders = ','.join(['?'] * len(valores))
    colunas = ','.join(valores.keys())
    cursor = conn.cursor()

    # Tenta inserir o registro. Se já existir (violando a constraint UNIQUE), a inserção é ignorada.
    cursor.execute(f"INSERT OR IGNORE INTO {tabela} ({colunas}) VALUES ({placeholders})", tuple(valores.values()))

    # Seleciona o ID (rowid) do registro, seja ele recém-inserido ou já existente.
    query = f"SELECT rowid FROM {tabela} WHERE " + ' AND '.join([f"{k}=?" for k in valores])
    cursor.execute(query, tuple(valores.values()))
    row = cursor.fetchone()

    return row[0] # Retorna o ID do registro

def carregar_dados(ticker: str, path_projeto: str) -> None:
    """
    Carrega dados transformados de um arquivo CSV para um banco de dados SQLite,
    otimizando a inserção e gerenciando as tabelas de dimensões e fatos.
    """
    print('\nIniciando a carga de dados...')

    # Define os caminhos para o arquivo CSV de entrada e o banco de dados SQLite de saída
    caminho_csv = os.path.join(path_projeto, f'data/transformed/{ticker.split(".")[0]}/dados_transformados_{ticker.split(".")[0]}.csv')
    caminho_db = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/SQLite_{ticker.split(".")[0]}.db')

    # Cria o diretório 'db' se ele não existir
    os.makedirs(os.path.dirname(caminho_db), exist_ok=True)

    # Verifica se o arquivo CSV de entrada existe
    if not os.path.exists(caminho_csv):
        print(f"[ERRO] Arquivo CSV não encontrado: {caminho_csv}")
        return

    # Carrega os dados do CSV para um DataFrame, parseando a coluna 'datetime' como tipo data/hora
    df = pd.read_csv(caminho_csv, parse_dates=['datetime'])

    # Verifica se o DataFrame carregado está vazio
    if df.empty:
        print("[INFO] Nenhum dado para inserir.")
        return

    # Conecta-se ao banco de dados e cria as tabelas se necessário
    conn = conectar_db(caminho_db)
    criar_tabelas(conn)

    print("Processando e inserindo dados nas tabelas de dimensão...")

    # Dicionários de cache para armazenar IDs de dimensões já processadas e evitar consultas repetidas
    cache_dim_tempo = {}
    cache_dim_indicadores = {}
    cache_dim_agregados = {}

    # Lista para coletar os valores das linhas da tabela de fatos para inserção em batch
    fato_valores_para_inserir = []

    # Itera sobre cada linha do DataFrame para processar e coletar dados
    for index, row in df.iterrows():
        # Formata as colunas 'data', 'hora' e 'datetime' como strings para inserção no SQLite
        data_str = row['data']
        hora_str = row['hora']
        datetime_str = row['datetime'].isoformat() # Converte Timestamp para string ISO 8601

        # Processa e obtém o ID para a dimensão de tempo
        valores_tempo_tuple = (data_str, hora_str, datetime_str, row['dia_semana'])
        if valores_tempo_tuple not in cache_dim_tempo:
            id_tempo = obter_id(conn, 'dim_tempo', {
                'data': data_str,
                'hora': hora_str,
                'datetime': datetime_str,
                'dia_da_semana': row['dia_semana']
            })
            cache_dim_tempo[valores_tempo_tuple] = id_tempo
        else:
            id_tempo = cache_dim_tempo[valores_tempo_tuple]

        # Processa e obtém o ID para a dimensão de indicadores
        valores_ind_tuple = (row['RSI'], row['EMA_20'], row['retorno'])
        if valores_ind_tuple not in cache_dim_indicadores:
            id_ind = obter_id(conn, 'dim_indicadores', {
                'RSI': row['RSI'], 'EMA_20': row['EMA_20'], 'retorno': row['retorno']
            })
            cache_dim_indicadores[valores_ind_tuple] = id_ind
        else:
            id_ind = cache_dim_indicadores[valores_ind_tuple]

        # Processa e obtém o ID para a dimensão de agregados
        valores_agg_tuple = (
            row['hora_num'], row['minuto'],
            row['abertura_dia'], row['minima_dia'], row['maxima_dia'],
            row['fechamento_dia'], row['volume_dia']
        )
        if valores_agg_tuple not in cache_dim_agregados:
            id_agg = obter_id(conn, 'dim_agregados', {
                'hora_num': row['hora_num'], 'minuto': row['minuto'],
                'abertura_dia': row['abertura_dia'],
                'minima_dia': row['minima_dia'],
                'maxima_dia': row['maxima_dia'],
                'fechamento_dia': row['fechamento_dia'],
                'volume_dia': row['volume_dia']
            })
            cache_dim_agregados[valores_agg_tuple] = id_agg
        else:
            id_agg = cache_dim_agregados[valores_agg_tuple]

        # Adiciona os valores da linha de fato à lista para inserção em batch
        fato_valores_para_inserir.append((
            row['abertura_candle'], row['min_candle'], row['max_candle'], row['fechamento_candle'],
            row['volume_candle'],
            id_tempo, id_ind, id_agg
        ))

    # Realiza a inserção em batch para a tabela de fatos
    cursor = conn.cursor()
    try:
        cursor.executemany("""
            INSERT OR IGNORE INTO fato_precos (
                abertura, minima, maxima, fechamento, volume,
                id_tempo, id_indicadores, id_agregados
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, fato_valores_para_inserir)
        conn.commit() # Confirma a transação
        novos_registros = cursor.rowcount # Obtém o número de linhas realmente inseridas
    except sqlite3.Error as e:
        conn.rollback() # Em caso de erro, desfaz a transação
        print(f"[ERRO] Erro durante a inserção de fato_precos: {e}")
        novos_registros = 0

    conn.close() # Fecha a conexão com o banco de dados
    print(f"Carga concluída com {novos_registros} novos registros na tabela fato_precos.")

# Execução direta poara testes
if __name__ == "__main__":
    carregar_dados('LREN3', '/content/pregao_predict')
