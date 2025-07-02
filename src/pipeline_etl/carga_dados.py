
#@title SRC Carga de dados

import sqlite3
import pandas as pd
import os

def conectar_db(caminho_db):
    return sqlite3.connect(caminho_db)

def criar_tabela_simples(conn):
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dados_limpos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ativo TEXT NOT NULL,
        data TEXT NOT NULL,
        hora TEXT NOT NULL,
        abertura REAL NOT NULL,
        maximo REAL NOT NULL,
        minimo REAL NOT NULL,
        fechamento REAL NOT NULL,
        volume REAL NOT NULL,
        quantidade REAL NOT NULL,
        UNIQUE(ativo, data, hora)
    );
    """)
    conn.commit()

def carregar_dados(ticker: str, intervalo: str, path_projeto: str) -> None:
    print('\nIniciando a carga de dados (simples)...')

    caminho_csv = os.path.join(
        path_projeto,
        f'data/cleaned/{ticker.split(".")[0]}/{intervalo}/dados_limpos_{ticker.split(".")[0]}_{intervalo}.csv'
    )
    caminho_db = os.path.join(
        path_projeto,
        f'db/{ticker.split(".")[0]}/SQLite_{ticker.split(".")[0]}_{intervalo}.db'
    )

    if not os.path.exists(caminho_csv):
        print(f"[ERRO] Arquivo CSV não encontrado: {caminho_csv}")
        return

    os.makedirs(os.path.dirname(caminho_db), exist_ok=True)

    df = pd.read_csv(caminho_csv)

    if df.empty:
        print("[INFO] Nenhum dado para inserir.")
        return

    conn = conectar_db(caminho_db)
    criar_tabela_simples(conn)

    cursor = conn.cursor()
    novos_registros = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO dados_limpos (
                    ativo, data, hora, abertura, maximo, minimo,
                    fechamento, volume, quantidade
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['Ativo'], row['Data'], row['Hora'], row['Abertura'],
                row['Máximo'], row['Mínimo'], row['Fechamento'],
                row['Volume'], row['Quantidade']
            ))
            novos_registros += cursor.rowcount
        except Exception as e:
            print(f"[ERRO] Falha ao inserir linha: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"[✓] Carga concluída com {novos_registros} registros inseridos.")

# Execução direta para testes
if __name__ == "__main__":
    carregar_dados('LREN3', '5m', '/content/pregao_predict')
