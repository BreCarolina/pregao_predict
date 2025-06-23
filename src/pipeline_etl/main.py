
#@title Script principal para executar o pipeline ETL

import os
import pandas as pd
from src.pipeline_etl.extracao_dados import extrair_dados
from src.pipeline_etl.transformacao_dados_versao001 import transformar_dados
from src.pipeline_etl.carga_dados import carregar_dados
from src.pipeline_etl.gerar_catalogo_dados_v2 import gerar_catalogo_dados



def main(ticker: str, path_projeto: str, intervalo: str) -> None:
    """
    Executa o pipeline completo de ETL para dados de mercado de um ticker específico.

    Args:
        ticker (str): O ticker do ativo (ex: 'BBDC4.SA').
        path_projeto (str): O caminho base do diretório do projeto.
        intervalo (str): O intervalo dos dados (ex: '5m' para 5 minutos).
    """
    print("\nIniciando pipeline de dados...")

    # --- Extração de Dados ---
    print("\nIniciando etapa de Extração...")
    df_extraido = extrair_dados(ticker, intervalo, path_projeto)
    if df_extraido is None or df_extraido.empty:
        print("\n[ERRO] Falha na extração: nenhum dado retornado. Encerrando o pipeline.")
        return

    print(">> Extração concluída com sucesso.")

    # --- Transformação de Dados ---
    print("\nIniciando etapa de Transformação...")
    transformar_dados(ticker, path_projeto)
    print(">> Transformação concluída com sucesso.")

    # --- Carga de Dados ---
    print("\nIniciando etapa de Carga...")
    carregar_dados(ticker, path_projeto)
    print(">> Carga concluída com sucesso.")

    # --- Atualização/Geração do Catálogo ---
    print("\nIniciando atualização do catálogo de dados...")
    gerar_catalogo_dados(ticker.split('.')[0], path_projeto)
    print(">> Catálogo atualizado com sucesso.")

    print("\n>> Pipeline concluído com sucesso.")

# Execução direta para testes
if __name__ == "__main__":
    # Define as variáveis para a execução do pipeline
    ticker = 'BBDC4.SA'
    path_projeto = '/content/Piloto_day_trade'
    intervalo = '5m'

    # Chama a função principal do pipeline
    main(ticker, path_projeto, intervalo)
