
#@title Script principal para executar o pipeline ETL

import os
import pandas as pd
from src.pipeline_etl.extracao_dados import extrair_dados
from src.pipeline_etl.transformacao_dados import transformar_dados
from src.pipeline_etl.carga_dados import carregar_dados
from src.pipeline_etl.gerar_catalogo_dados import gerar_catalogo_dados

def main(ticker: str, intervalo: str, path_projeto: str) -> None:
  
    # --- Extração de Dados ---
    print("\nIniciando etapa de Extração...")
    df_extraido = extrair_dados(ticker, intervalo, path_projeto)
    if df_extraido is None or df_extraido.empty:
        print("\n[ERRO] Falha na extração: nenhum dado retornado. Encerrando o pipeline.")
        return
    print(">> Extração concluída com sucesso.")

    # --- Transformação de Dados ---
    print("\nIniciando etapa de Transformação...")
    transformar_dados(ticker, intervalo, path_projeto)
    print(">> Transformação concluída com sucesso.")

    # --- Carga de Dados ---
    print("\nIniciando etapa de Carga...")
    carregar_dados(ticker, intervalo, path_projeto)
    print(">> Carga concluída com sucesso.")

    # --- Atualização/Geração do Catálogo ---
    print("\nIniciando atualização do catálogo de dados...")
    gerar_catalogo_dados(ticker, intervalo, path_projeto)
    print(">> Catálogo atualizado com sucesso.")    

# Execução direta para testes
if __name__ == "__main__":
    # Define as variáveis para a execução do pipeline
    ticker = 'LREN3.SA'
    intervalo = '15m'
    path_projeto = '/content/pregao_predict'   

    # Chama a função principal do pipeline
    main(ticker, intervalo, path_projeto)
    print("\n>> Pipeline concluído com sucesso.")
