
import sqlite3
import pandas as pd
import os

def conectar_db(caminho_db):
    """Conecta ou cria o banco de dados SQLite."""
    return sqlite3.connect(caminho_db)

def criar_tabelas(conn):
    """Cria a tabela de tempo e fato_precos no banco de dados se ainda não existirem."""
    cursor = conn.cursor()

    # Tabela de dimensão tempo
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_tempo (
        id_tempo INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT NOT NULL,
        hora TEXT NOT NULL,
        datetime TEXT NOT NULL UNIQUE,
        dia_da_semana INTEGER NOT NULL
    );
    """)

    # Tabela fato com retorno e volatilidade
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fato_precos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        abertura REAL NOT NULL,
        minima REAL NOT NULL,
        maxima REAL NOT NULL,
        fechamento REAL NOT NULL,
        volume INTEGER NOT NULL,
        retorno REAL NOT NULL,
        volatilidade REAL NOT NULL,
        id_tempo INTEGER NOT NULL,
        FOREIGN KEY (id_tempo) REFERENCES dim_tempo(id_tempo),
        UNIQUE(id_tempo)
    );
    """)

    conn.commit()

def obter_id_tempo(conn, data, hora, datetime_str, dia_da_semana):
    """Obtém ou insere a dimensão de tempo e retorna seu ID."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO dim_tempo (data, hora, datetime, dia_da_semana)
        VALUES (?, ?, ?, ?)
    """, (data, hora, datetime_str, dia_da_semana))
    
    cursor.execute("""
        SELECT id_tempo FROM dim_tempo
        WHERE datetime = ?
    """, (datetime_str,))
    
    row = cursor.fetchone()
    return row[0] if row else None

def carregar_dados(ticker: str, intervalo: str, path_projeto: str) -> None:
    print('\nIniciando a carga de dados...')
    
    caminho_csv = os.path.join(path_projeto, f'data/transformed/{ticker.split(".")[0]}/{intervalo}/dados_transformados_{ticker.split(".")[0]}_{intervalo}.csv')
    caminho_db = os.path.join(path_projeto, f'db/{ticker}/SQLite_{ticker.split(".")[0]}_{intervalo}.db')

    if not os.path.exists(caminho_csv):
        print(f"[ERRO] Arquivo CSV não encontrado: {caminho_csv}")
        return

    os.makedirs(os.path.dirname(caminho_db), exist_ok=True)

    df = pd.read_csv(caminho_csv, parse_dates=['datetime'])

    if df.empty:
        print("[INFO] Nenhum dado para inserir.")
        return

    conn = conectar_db(caminho_db)
    criar_tabelas(conn)

    print("Processando e inserindo dados...")

    cursor = conn.cursor()
    novos_registros = 0

    for _, row in df.iterrows():
        data = row['data']
        hora = row['hora']
        datetime_str = row['datetime'].isoformat()
        dia_semana = row['dia_semana']

        id_tempo = obter_id_tempo(conn, data, hora, datetime_str, dia_semana)
        if id_tempo is None:
            continue

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO fato_precos (
                    abertura, minima, maxima, fechamento, volume,
                    retorno, volatilidade, id_tempo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['abertura'], row['minimo'], row['maximo'], row['fechamento'],
                row['volume'], row['retorno'], row['volatilidade'], id_tempo
            ))
            novos_registros += cursor.rowcount
        except Exception as e:
            print(f"[ERRO] Falha ao inserir dados: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"[✓] Carga concluída com {novos_registros} novos registros na tabela fato_precos.")

# Execução direta para testes
if __name__ == "__main__":
    carregar_dados('BBDC4', '15m', '/content/pregao_predict')
