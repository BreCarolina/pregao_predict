
#@title Formatar/limpar  dados do Profit
import os
import pandas as pd

def formatar_dados_profit( ticker, intervalo, path_raw, path_projeto):
    # Define o caminho dos dados limpos (Saida)
    path_cleaned = os.path.join(path_projeto, f'data/cleaned/{ticker}/{intervalo}/dados_limpos_{ticker}_{intervalo}.csv')
    os.makedirs(os.path.dirname(path_cleaned), exist_ok=True)

    # Lê o CSV com separador correto e encoding adequado
    df = pd.read_csv(
        path_raw,
        encoding='latin1',
        sep=';',
        dtype=str  # Lê tudo como string para limpeza manual
    )

    # Remove espaços, tabulações e caracteres não visíveis nas strings
    df = df.applymap(lambda x: x.replace('\t', '').replace('\xa0', '').strip() if isinstance(x, str) else x)

    # Converte colunas numéricas, tratando pontos e vírgulas
    colunas_numericas = ['Abertura', 'Máximo', 'Mínimo', 'Fechamento', 'Volume', 'Quantidade']
    for col in colunas_numericas:
        df[col] = df[col].str.replace('.', '', regex=False)  # Remove pontos de milhar
        df[col] = df[col].str.replace(',', '.', regex=False)  # Troca vírgula por ponto decimal
        df[col] = pd.to_numeric(df[col], errors='coerce')    # Converte para numérico (float)

    # Corrige formatação da coluna 'Hora' para padrão datetime.time
    df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M:%S', errors='coerce').dt.time

    # Converte 'Data' para datetime, formato dia primeiro
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')

    # Remove registros da data 30/08/2024 (se quiser remover essa data)
    data_remover = pd.to_datetime('30/08/2024', dayfirst=True)
    df = df[df['Data'] != data_remover]

    # Ordena pelo par Data e Hora
    df = df.sort_values(by=['Data', 'Hora'], ascending=[True, True])

    # Salva CSV limpo (sem índice)
    df.to_csv(path_cleaned, index=False)

    # Exibe as primeiras linhas para conferência
    print(f"\n[INFO] Dados limpos salvos em: {path_cleaned}")
    print(df.head(10))

    return df

if __name__ == "__main__":
    ticker = 'LREN3'
    intervalo = '5m'
    path_projeto = '/content/pregao_predict'
    path_raw = '/content/LREN3_B_0_5min.csv'
    formatar_dados_profit(ticker, intervalo, path_raw, path_projeto)
