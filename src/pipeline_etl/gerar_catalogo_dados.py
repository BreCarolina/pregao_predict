
#@title SRC Gerar Catálogo de dados

import os
import sqlite3
import pandas as pd
from datetime import datetime

def gerar_catalogo_dados(ticker: str, intervalo: str, path_projeto: str):
    print(f"\n--- Gerando catálogo de dados para {ticker} ---")

    caminho_db = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/SQLite_{ticker.split(".")[0]}_{intervalo}.db')
    caminho_catalogo = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/catalogo_dados_{ticker.split(".")[0]}_{intervalo}.md')

    if not os.path.exists(caminho_db):
        print(f"[ERRO] Banco de dados não encontrado em: {caminho_db}.")
        return

    os.makedirs(os.path.dirname(caminho_catalogo), exist_ok=True)

    descricoes_colunas = {
        'id': 'Chave primária da tabela.',
        'ativo': 'Código do ativo negociado.',
        'data': 'Data da negociação (YYYY-MM-DD).',
        'hora': 'Hora da negociação (HH:MM:SS).',
        'abertura': 'Preço de abertura do candle.',
        'maximo': 'Preço máximo do candle.',
        'minimo': 'Preço mínimo do candle.',
        'fechamento': 'Preço de fechamento do candle.',
        'volume': 'Volume negociado no candle.',
        'quantidade': 'Quantidade de negócios no candle.'
    }

    try:
        conn = sqlite3.connect(caminho_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tabelas = [t[0] for t in cursor.fetchall() if not t[0].startswith('sqlite_')]

        with open(caminho_catalogo, 'w', encoding='utf-8') as f:
            f.write(f"# Catálogo de Dados - {ticker.split('.')[0]}\n\n")
            f.write(f"**Atualizado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            for tabela in sorted(tabelas):
                f.write(f"## Tabela: `{tabela}`\n\n")
                f.write("| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |\n")
                f.write("|---|---|---|---|---|\n")

                cursor.execute(f"PRAGMA table_info('{tabela}');")
                colunas = cursor.fetchall()

                for col in colunas:
                    _, nome, tipo, notnull, _, pk = col
                    descricao = descricoes_colunas.get(nome, "-")
                    f.write(f"| `{nome}` | `{tipo}` | {'Não' if notnull else 'Sim'} | {'Sim' if pk else 'Não'} | {descricao} |\n")

                f.write("\n")

            f.write("---\n\n")
            f.write("## Observações\n\n")
            f.write("- A base contém apenas a tabela `dados_limpos`, com dados de candles de 5 minutos.\n")
            f.write("- Cada linha representa um momento único da negociação (ativo + data + hora).\n")
            f.write("- A estrutura é simples e direta, ideal para análises e modelagens temporais.\n")

        print(f"[✓] Catálogo atualizado com sucesso: {caminho_catalogo}")

    except Exception as e:
        print(f"[ERRO] Falha ao gerar catálogo: {e}")
    finally:
        if conn:
            conn.close()
            print("[✓] Conexão com o banco encerrada.")

# Execução direta
if __name__ == "__main__":
    ticker = 'LREN3'
    intervalo = '5min'
    path_projeto = '/content/pregao_predict'
    gerar_catalogo_dados(ticker, intervalo, path_projeto)
