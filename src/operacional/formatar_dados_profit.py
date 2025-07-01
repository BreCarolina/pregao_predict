
#@title Formatar/limpar  dados do Profit

import pandas as pd

def formatar_dados_profit():
    # Lê o CSV com separador correto e encoding adequado
    df = pd.read_csv(
        '/content/pregao_predict/data/raw/LREN3_B_0_5min.csv',
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
    df.to_csv('/content/pregao_predict/data/cleaned/LREN3_5min_limpos.csv', index=False)
    
    # Exibe as primeiras linhas para conferência
    print(df.head(10))
    
    return df
