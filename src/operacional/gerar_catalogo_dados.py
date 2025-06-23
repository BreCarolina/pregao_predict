
import sqlite3
import os
import pandas as pd
from datetime import datetime 

def gerar_catalogo_dados(ticker: str, path_projeto: str):
    """
    Gera um catálogo de dados em formato Markdown para o banco de dados SQLite.
    O catálogo inclui informações sobre tabelas, colunas, tipos de dados e descrições.
    """
    print(f"\n--- Gerando catálogo de dados para {ticker} ---")

    caminho_db = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/SQLite_{ticker.split(".")[0]}.db')
    caminho_catalogo = os.path.join(path_projeto, f'db/{ticker.split(".")[0]}/catalogo_dados_{ticker.split(".")[0]}.md')

    # Garante que o diretório 'docs' existe
    os.makedirs(os.path.dirname(caminho_catalogo), exist_ok=True)

    if not os.path.exists(caminho_db):
        print(f"[ERRO] Banco de dados não encontrado em: {caminho_db}. Não é possível gerar o catálogo.")
        return

    conn = None
    try:
        conn = sqlite3.connect(caminho_db)
        cursor = conn.cursor()

        # Obtém informações sobre as tabelas no banco de dados
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tabelas = cursor.fetchall()

        with open(caminho_catalogo, 'w', encoding='utf-8') as f:
            f.write(f"# Catálogo de Dados - {ticker.split('.')[0]}\n\n")
            f.write(f"Este documento descreve a estrutura e o conteúdo do banco de dados SQLite para o ticker **{ticker}**.\n\n")
            f.write(f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n") # Usar datetime.now()
            f.write("---\n\n")

            for tabela in tabelas:
                nome_tabela = tabela[0]
                if nome_tabela.startswith('sqlite_'): # Ignora tabelas internas do SQLite
                    continue

                f.write(f"## Tabela: `{nome_tabela}`\n\n")

                # Consulta para obter o schema da tabela
                cursor.execute(f"PRAGMA table_info('{nome_tabela}');")
                colunas = cursor.fetchall()

                f.write("| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |\n")
                f.write("|---|---|---|---|---|\n")

                # Dicionário de descrições predefinidas para colunas comuns
                # Adapte estas descrições de acordo com o seu domínio e as suas colunas
                descricoes_colunas = {
                    # dim_tempo
                    'id_tempo': 'Identificador único da dimensão de tempo.',
                    'data': 'Data do candle (YYYY-MM-DD).',
                    'hora': 'Hora do candle (HH:MM:SS).',
                    'datetime': 'Timestamp completo do candle (YYYY-MM-DD HH:MM:SS.ms).',
                    'dia_da_semana': 'Dia da semana (0=Segunda, 6=Domingo).',

                    # dim_indicadores
                    'id_indicadores': 'Identificador único da dimensão de indicadores.',
                    'RSI': 'Índice de Força Relativa (Relative Strength Index).',
                    'EMA_20': 'Média Móvel Exponencial de 20 períodos.',
                    'retorno': 'Retorno percentual do candle atual.',

                    # dim_agregados
                    'id_agregados': 'Identificador único da dimensão de agregados diários.',
                    'hora_num': 'Componente da hora do dia (0-23).',
                    'minuto': 'Componente do minuto do dia (0-59).',
                    'abertura_dia': 'Preço de abertura do dia.',
                    'minima_dia': 'Preço mínimo do dia.',
                    'maxima_dia': 'Preço máximo do dia.',
                    'fechamento_dia': 'Preço de fechamento do dia.',
                    'volume_dia': 'Volume total de negociação do dia.',

                    # fato_precos
                    'id': 'Chave primária da tabela de fatos.',
                    'abertura': 'Preço de abertura do candle.',
                    'minima': 'Preço mínimo do candle.',
                    'maxima': 'Preço máximo do candle.',
                    'fechamento': 'Preço de fechamento do candle.',
                    'volume': 'Volume de negociação do candle.',
                    'id_tempo': 'Chave estrangeira para a dimensão de tempo.',
                    'id_indicadores': 'Chave estrangeira para a dimensão de indicadores.',
                    'id_agregados': 'Chave estrangeira para a dimensão de agregados diários.',

                    # Outras colunas que possam aparecer
                    'is_outlier': 'Indicador binário se o candle é considerado um outlier.',
                    'RSI_indicador_faltante': 'Indicador binário se o RSI original era NaN.',
                    'EMA_20_indicador_faltante': 'Indicador binário se a EMA_20 original era NaN.',
                    'abertura_candle': 'Preço de abertura do candle.',
                    'min_candle': 'Preço mínimo do candle.',
                    'max_candle': 'Preço máximo do candle.',
                    'fechamento_candle': 'Preço de fechamento do candle.',
                    'volume_candle': 'Volume de negociação do candle.',
                }

                for col in colunas:
                    cid, nome, tipo, notnull, dflt_value, pk = col
                    
                    # Usa a descrição predefinida ou uma genérica
                    descricao = descricoes_colunas.get(nome, f"Coluna {nome} da tabela {nome_tabela}.")

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

        print(f"[✓] Catálogo de dados gerado com sucesso em: {caminho_catalogo}")

    except sqlite3.Error as e:
        print(f"[ERRO] Erro ao conectar ou consultar o banco de dados: {e}")
    finally:
        if conn:
            conn.close()

# Execução direta (testes)
if __name__ == '__main__':    
    ticker_alvo = 'LREN3' 
    caminho_base_projeto = '/content/Piloto_day_trade'    
    gerar_catalogo_dados(ticker_alvo, caminho_base_projeto)
    print("\nProcesso de geração de catálogo de dados concluído.")
