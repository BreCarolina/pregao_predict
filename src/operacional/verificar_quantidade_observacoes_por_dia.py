#@title Verificar quantidade de observações por dia

def verificar_quantidade_observacoes_por_dia(df):
    print("[INFO] Verificando quantidade de observações por dia...")

    # Garante que a coluna 'Data' está no formato datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Agrupa por dia e conta número de linhas por grupo
    contagem_por_dia = df.groupby(df['Data'].dt.date).size()

    # Frequência de cada quantidade de candles
    freq = contagem_por_dia.value_counts().sort_index()

    print("\n[INFO] Frequência das quantidades de candles por dia:")
    for qtd, n_ocorrencias in freq.items():
        print(f"- {qtd} candles: {n_ocorrencias} dias")

    # Identifica a quantidade predominante
    qtd_predominante = freq.idxmax()
    print(f"\n[INFO] Quantidade de candles mais comum: {qtd_predominante}")

    # Identifica dias fora do padrão
    dias_diferentes = contagem_por_dia[contagem_por_dia != qtd_predominante]

    if not dias_diferentes.empty:
        print("\n[AVISO] Dias com quantidade diferente de candles:")
        for data, qtd in dias_diferentes.items():
            print(f"- {data}: {qtd} candles")
    else:
        print("\n[INFO] Todos os dias possuem a mesma quantidade de candles.")

if __name__ == "__main__":
    df_limpo  = pd.read_csv('/content/pregao_predict/data/cleaned/LREN3/5m/dados_limpos_LREN3_5m.csv')
    verificar_quantidade_observacoes_por_dia(df_limpo)
