
import pandas as pd
import numpy as np
import os
import glob

def diagnostico_dados(df: pd.DataFrame, pasta_relatorios: str = "reports") -> dict:
    resultados = {}
    data_base = df['datetime'].dt.date.max().strftime("%Y-%m-%d")
    nome_arquivo = f"diagnostico_consistencia_dados_{data_base}.txt"
    caminho_pasta = os.path.join(os.getcwd(), pasta_relatorios)
    caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo)

    if not os.path.exists(caminho_pasta):
        os.makedirs(caminho_pasta)

    linhas = []
    linhas.append(f"Relatório de Diagnóstico - Dados de {data_base}\n")
    linhas.append("Verificações de Consistência:\n")

    # 1. Ausência de nulos
    nulos = df.isnull().sum()
    for col, val in nulos.items():
        if val > 0:
            linhas.append(f"Aviso: {val} valores nulos em '{col}'.\n")
        else:
            linhas.append(f"OK: Coluna '{col}' sem nulos.\n")
    resultados['nulos'] = nulos.to_dict()

    # 2. Consistência de volume
    volume = df.get('volume')
    if volume is not None:
        negativos = (volume < 0).sum()
        zeros = (volume == 0).sum()
        if negativos > 0:
            linhas.append(f"Erro: {negativos} valores negativos em volume.\n")
        else:
            linhas.append("OK: Nenhum volume negativo.\n")
        if zeros > 0:
            linhas.append(f"Aviso: {zeros} volumes zerados.\n")
        else:
            linhas.append("OK: Nenhum volume zero.\n")
        resultados['volume_negativo'] = negativos
        resultados['volume_zero'] = zeros

    # 3. Tipos corretos
    tipos_incorretos = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
        if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype))
    }
    if tipos_incorretos:
        for col, tipo in tipos_incorretos.items():
            linhas.append(f"Erro: Tipo incorreto na coluna '{col}' ({tipo}).\n")
    else:
        linhas.append("OK: Todos os tipos de dados estão corretos.\n")
    resultados['tipos_incorretos'] = tipos_incorretos

    # 4. Ausência de duplicatas
    duplicatas = df.duplicated().sum()
    if duplicatas > 0:
        linhas.append(f"Aviso: {duplicatas} linhas duplicadas.\n")
    else:
        linhas.append("OK: Nenhuma duplicata.\n")
    resultados['duplicatas'] = duplicatas

    # 5. Consistência temporal
    if 'datetime' in df.columns:
        df_sorted = df.sort_values('datetime')
        ordem_ok = df_sorted['datetime'].is_monotonic_increasing
        linhas.append("OK: Ordem temporal correta.\n" if ordem_ok else "Erro: Dados fora de ordem temporal.\n")
        resultados['ordem_temporal'] = ordem_ok

        diffs = df_sorted['datetime'].diff().value_counts().to_dict()
        resultados['intervalos_temporais'] = {str(k): int(v) for k, v in diffs.items()}
    else:
        linhas.append("Erro: Coluna 'datetime' ausente.\n")
        resultados['ordem_temporal'] = False
        resultados['intervalos_temporais'] = {}

    # 6. Outliers extremos (z-score > 5)
    df_numeric = df.select_dtypes(include=[np.number])
    zscores = (df_numeric - df_numeric.mean()) / df_numeric.std()
    outliers = (np.abs(zscores) > 5).sum()
    for col, qtde in outliers.items():
        if qtde > 0:
            linhas.append(f"Aviso: {qtde} outliers extremos em '{col}'.\n")
    resultados['outliers'] = outliers.to_dict()

    # 7. Checagem dos indicadores
    indicadores_check = {}
    fechamento = df.get('fechamento')
    if fechamento is not None:
        if 'SMA_10' in df.columns:
            sma_teorico = fechamento.rolling(10).mean()
            match_pct = np.isclose(df['SMA_10'], sma_teorico, rtol=1e-3, atol=1e-4).mean()
            indicadores_check['SMA_10'] = match_pct
            linhas.append(f"SMA_10: {match_pct:.2%} dos valores coerentes.\n")
        if 'EMA_10' in df.columns:
            ema_teorico = fechamento.ewm(span=10, adjust=False).mean()
            match_pct = np.isclose(df['EMA_10'], ema_teorico, rtol=1e-3, atol=1e-4).mean()
            indicadores_check['EMA_10'] = match_pct
            linhas.append(f"EMA_10: {match_pct:.2%} dos valores coerentes.\n")
    resultados['checagem_indicadores'] = indicadores_check

    # 8. Verificação de coerência entre features derivadas
    coerencia = {}
    if 'retorno_log' in df.columns and fechamento is not None:
        retorno_teorico = np.log(fechamento / fechamento.shift(1))
        match_pct = np.isclose(df['retorno_log'], retorno_teorico, rtol=1e-4, atol=1e-6).mean()
        coerencia['retorno_log'] = match_pct
        linhas.append(f"retorno_log: {match_pct:.2%} dos valores coerentes.\n")
    if 'volatilidade' in df.columns and 'retorno_log' in df.columns:
        vol_teorica = df['retorno_log'].rolling(window=10).std()
        match_pct = np.isclose(df['volatilidade'], vol_teorica, rtol=1e-3, atol=1e-6).mean()
        coerencia['volatilidade'] = match_pct
        linhas.append(f"volatilidade: {match_pct:.2%} dos valores coerentes.\n")
    resultados['coerencia_features'] = coerencia

    # Impressão em tela
    print("".join(linhas))

    # Salvamento do relatório
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.writelines(linhas)

    # Gerenciamento: manter apenas os 10 arquivos mais recentes
    arquivos = sorted(glob.glob(os.path.join(caminho_pasta, "diagnostico_consistencia_dados_*.txt")),
                      key=os.path.getmtime)
    while len(arquivos) > 10:
        os.remove(arquivos[0])
        arquivos.pop(0)

    return resultados

if __name__ == "__main__":
    path_dados_transformados = '/content/Piloto_day_trade/data/transformed/dados_transformados_BBDC4.csv'
    df = pd.read_csv(path_dados_transformados, parse_dates=['datetime'])
    diagnostico_dados(df)

