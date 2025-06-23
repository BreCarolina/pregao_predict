
#@title Script Diagnóstico de consistência dos dados limpos

import pandas as pd
import os
from datetime import datetime

def diagnostico_dados_iniciais(df: pd.DataFrame, n=20, salvar_em_txt=True):
    """
    Executa diagnósticos sobre os primeiros registros do dataset.

    Parâmetros:
        df (pd.DataFrame): DataFrame com colunas ['data', 'hora', 'abertura', 'minimo', 'maximo', 'fechamento', 'volume', 'datetime']
        n (int): Quantidade de registros iniciais a avaliar
        salvar_em_txt (bool): Se True, salva relatório em reports/
    """
    relatorio = []

    relatorio.append("=== Diagnóstico Inicial dos Dados ===\n")

    # 1. Informações gerais
    relatorio.append(f"Total de registros: {len(df)}")
    relatorio.append(f"Período da série: {df['datetime'].min()} até {df['datetime'].max()}")
    relatorio.append(f"Frequência estimada: {df['datetime'].diff().mode()[0]}")

    # 2. Checagem de valores ausentes
    relatorio.append("\nValores nulos nos primeiros registros:")
    relatorio.append(str(df.head(n).isnull().sum()))

    # 3. Verificação de sequência temporal
    relatorio.append("\nIntervalos de tempo nos primeiros registros:")
    relatorio.append(str(df['datetime'].head(n).diff().value_counts(dropna=False)))

    # 4. Estatísticas de preços
    primeiros = df.head(n)
    relatorio.append("\nEstatísticas dos preços nos primeiros registros:")
    relatorio.append(str(primeiros[['abertura', 'minimo', 'maximo', 'fechamento']].describe()))

    relatorio.append("\nVolumes nos primeiros registros:")
    relatorio.append(str(primeiros['volume'].describe()))

    # Volumes zerados
    volumes_zerados = primeiros[primeiros['volume'] == 0]
    if not volumes_zerados.empty:
        relatorio.append(f"\nAtenção: {len(volumes_zerados)} registros com volume 0 nos primeiros {n} registros:")
        relatorio.append(str(volumes_zerados[['datetime', 'volume']]))

    # Inconsistência entre colunas
    if not pd.to_datetime(df['data'] + ' ' + df['hora']).equals(df['datetime']):
        relatorio.append("\nInconsistência entre 'data' + 'hora' e coluna 'datetime'. Verifique geração dos dados.")

    relatorio.append("\n=== Fim do Diagnóstico ===")

    # Impressão no console
    print("\n".join(relatorio))

    # Salvar relatório em arquivo txt
    if salvar_em_txt:
        os.makedirs("reports", exist_ok=True)
        data_execucao = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_arquivo = f"data/cleaned/diagnostico_inicial_{data_execucao}.txt"
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write("\n".join(relatorio))

if __name__ == "__main__":
    df_limpo= pd.read_csv('/content/pregao_predict/data/transformed/dados_transformados_BBDC4.csv', parse_dates=['datetime'])
    diagnostico_dados_iniciais(df_limpo)
