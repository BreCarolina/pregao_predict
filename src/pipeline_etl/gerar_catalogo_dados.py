
#@title Script para gerar o catálogo do banco de dados

import os
import re
import sqlite3
import pandas as pd
from datetime import datetime

def ler_catalogo_existente(caminho_catalogo):
    """
    Lê o arquivo markdown e extrai estrutura atual:
    Retorna dict: {tabela: set(colunas)}
    """
    if not os.path.exists(caminho_catalogo):
        return {}

    estrutura = {}
    tabela_atual = None

    with open(caminho_catalogo, 'r', encoding='utf-8') as f:
        for linha in f:
            match_tabela = re.match(r"^## Tabela: `(.+?)`", linha)
            if match_tabela:
                tabela_atual = match_tabela.group(1)
                estrutura[tabela_atual] = set()
                continue

            if tabela_atual and linha.startswith('| `'):
                cols = linha.split('|')
                if len(cols) > 1:
                    nome_coluna = cols[1].strip().strip('`')
                    estrutura[tabela_atual].add(nome_coluna)

    return estrutura

def gerar_catalogo_dados(ticker: str, intervalo: str, path_projeto: str):
    print(f"\n--- Gerando catálogo de dados para {ticker} ---")

    caminho_db = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/SQLite_{ticker.split(".")[0]}_{intervalo}.db')
    caminho_catalogo = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/catalogo_dados_{ticker.split(".")[0]}_{intervalo}.md')

    os.makedirs(os.path.dirname(caminho_catalogo), exist_ok=True)

    if not os.path.exists(caminho_db):
        print(f"[ERRO] Banco de dados não encontrado em: {caminho_db}. Não é possível gerar o catálogo.")
        return

    catalogo_atual = ler_catalogo_existente(caminho_catalogo)

    conn = None
    try:
        conn = sqlite3.connect(caminho_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tabelas_db = [t[0] for t in cursor.fetchall() if not t[0].startswith('sqlite_')]

        tabelas_colunas_db = {}
        for tabela in tabelas_db:
            cursor.execute(f"PRAGMA table_info('{tabela}');")
            colunas = cursor.fetchall()
            colunas_nome = set([col[1] for col in colunas])
            tabelas_colunas_db[tabela] = colunas_nome

        tabelas_para_remover = set(catalogo_atual.keys()) - set(tabelas_db)
        for tabela in tabelas_para_remover:
            catalogo_atual.pop(tabela)

        tabelas_novas = set(tabelas_db) - set(catalogo_atual.keys())
        for tabela in tabelas_novas:
            catalogo_atual[tabela] = set()

        for tabela in tabelas_db:
            colunas_db = tabelas_colunas_db[tabela]
            colunas_atuais = catalogo_atual.get(tabela, set())
            colunas_removidas = colunas_atuais - colunas_db
            catalogo_atual[tabela] = (colunas_atuais - colunas_removidas) | (colunas_db - colunas_atuais)

        with open(caminho_catalogo, 'w', encoding='utf-8') as f:
            f.write(f"# Catálogo de Dados - {ticker.split('.')[0]}\n\n")
            f.write(f"Este documento descreve a estrutura e o conteúdo do banco de dados SQLite para o ticker **{ticker}**.\n\n")
            f.write(f"**Atualizado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            descricoes_colunas = {
                'id_tempo': 'Identificador único da dimensão de tempo.',
                'data': 'Data do candle (YYYY-MM-DD).',
                'hora': 'Hora do candle (HH:MM:SS).',
                'datetime': 'Timestamp completo do candle (YYYY-MM-DD HH:MM:SS.ms).',
                'dia_da_semana': 'Dia da semana (0=Segunda, 6=Domingo).',
                'id_indicadores': 'Identificador único da dimensão de indicadores.',
                'RSI': 'Índice de Força Relativa (Relative Strength Index).',
                'EMA_20': 'Média Móvel Exponencial de 20 períodos.',
                'retorno': 'Retorno percentual do candle atual.',
                'id_agregados': 'Identificador único da dimensão de agregados diários.',
                'hora_num': 'Componente da hora do dia (0-23).',
                'minuto': 'Componente do minuto do dia (0-59).',
                'abertura_dia': 'Preço de abertura do dia.',
                'minima_dia': 'Preço mínimo do dia.',
                'maxima_dia': 'Preço máximo do dia.',
                'fechamento_dia': 'Preço de fechamento do dia.',
                'volume_dia': 'Volume total de negociação do dia.',
                'id': 'Chave primária da tabela de fatos.',
                'abertura': 'Preço de abertura do candle.',
                'minima': 'Preço mínimo do candle.',
                'maxima': 'Preço máximo do candle.',
                'fechamento': 'Preço de fechamento do candle.',
                'volume': 'Volume de negociação do candle.',
                'id_tempo': 'Chave estrangeira para a dimensão de tempo.',
                'is_outlier': 'Indicador binário se o candle é considerado um outlier.',
                'RSI_indicador_faltante': 'Indicador binário se o RSI original era NaN.',
                'EMA_20_indicador_faltante': 'Indicador binário se a EMA_20 original era NaN.',
                'abertura_candle': 'Preço de abertura do candle.',
                'min_candle': 'Preço mínimo do candle.',
                'max_candle': 'Preço máximo do candle.',
                'fechamento_candle': 'Preço de fechamento do candle.',
                'volume_candle': 'Volume de negociação do candle.',
            }

            for tabela in sorted(catalogo_atual.keys()):
                f.write(f"## Tabela: `{tabela}`\n\n")
                f.write("| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |\n")
                f.write("|---|---|---|---|---|\n")

                cursor.execute(f"PRAGMA table_info('{tabela}');")
                colunas = cursor.fetchall()

                for col in colunas:
                    cid, nome, tipo, notnull, dflt_value, pk = col
                    if nome not in catalogo_atual[tabela]:
                        continue
                    descricao = descricoes_colunas.get(nome, f"Coluna {nome} da tabela {tabela}.")
                    f.write(f"| `{nome}` | `{tipo}` | {'Não' if notnull else 'Sim'} | {'Sim' if pk else 'Não'} | {descricao} |\n")
                f.write("\n")

            f.write("---\n\n")
            f.write("## Relacionamentos entre Tabelas\n\n")
            f.write("Este esquema utiliza um modelo estrela (Star Schema):\n\n")
            f.write("- **`fato_precos`** (Tabela de Fatos):\n")
            f.write("    - Contém as métricas de preço e volume por candle.\n")
            f.write("    - Relaciona-se com as tabelas de dimensão através das seguintes chaves estrangeiras:\n")
            f.write("        - `id_tempo` -> `dim_tempo.id_tempo`\n")
            f.write("        - `id_indicadores` -> `dim_indicadores.id_indicadores`\n")
            f.write("        - `id_agregados` -> `dim_agregados.id_agregados`\n\n")
            f.write("- **`dim_tempo`** (Dimensão de Tempo):\n")
            f.write("    - Descreve a granularidade temporal dos candles (data, hora, dia da semana).\n")
            f.write("    - Chave Primária: `id_tempo`\n\n")
            f.write("- **`dim_indicadores`** (Dimensão de Indicadores):\n")
            f.write("    - Armazena os valores dos indicadores técnicos (RSI, EMA_20, retorno).\n")
            f.write("    - Chave Primária: `id_indicadores`\n\n")
            f.write("- **`dim_agregados`** (Dimensão de Agregados Diários):\n")
            f.write("    - Contém os valores agregados diários (abertura do dia, mínima, máxima, fechamento, volume).\n")
            f.write("    - Chave Primária: `id_agregados`\n\n")
            f.write("---\n\n")
            f.write("## Observações Adicionais\n\n")
            f.write("- Os dados são armazenados na granularidade de candles de 5 minutos, conforme o intervalo de extração.\n")
            f.write("- As tabelas de dimensão são projetadas para reduzir a redundância de dados.\n")
            f.write("- Novos dados são inseridos incrementalmente, evitando duplicatas com base nas chaves de unicidade.\n")
            f.write("- Valores `NaN` para indicadores são tratados com interpolação linear, e um flag binário (`_indicador_faltante`) é adicionado para indicar sua ocorrência original.\n")
            f.write("- A coluna `is_outlier` indica a presença de valores extremos (outliers) no candle.\n")
            f.write("- `volume_candle` para candles preenchidos (lacunas) é definido como 0.\n")
            f.write("- Os agregados diários (`abertura_dia`, `minima_dia`, etc.) representam os valores do dia *daquele candle*.\n")
            f.write("\n")

        print(f"[✓] Catálogo de dados atualizado com sucesso em: {caminho_catalogo}")

        # Impressão da amostra resumida do catálogo atualizado
        try:
            with open(caminho_catalogo, 'r', encoding='utf-8') as f:
                linhas = f.readlines()

            print("\n--- Amostra resumida do catálogo atualizado ---\n")

            tabela_atual = None
            linhas_mostradas = 0
            max_linhas = 30

            for linha in linhas:
                if linhas_mostradas >= max_linhas:
                    break

                if linha.startswith("## Tabela:"):
                    tabela_atual = linha.strip()
                    print(tabela_atual)
                    linhas_mostradas += 1
                    continue

                if tabela_atual and linha.startswith("| `"):
                    partes = linha.split("|")
                    if len(partes) >= 3:
                        nome_col = partes[1].strip()
                        tipo_col = partes[2].strip()
                        print(f"  - Coluna: {nome_col} | Tipo: {tipo_col}")
                        linhas_mostradas += 1

            print("\n---")

        except Exception as e:
            print(f"[ERRO] Falha ao ler o catálogo para amostra: {e}")

    except sqlite3.Error as e:
        print(f"[ERRO] Erro ao conectar ou consultar o banco de dados: {e}")
    finally:
        if conn:
            conn.close()
            print("[✓] Conexão com o banco de dados fechada.")

if __name__ == "__main__":
    ticker = 'BBDC4'
    intervalo = '15m'
    path_projeto = '/content/pregao_predict'
    gerar_catalogo_dados(ticker, intervalo, path_projeto)
