
import pandas as pd
import plotly.graph_objects as go

def plotar_grafico_candles(df: pd.DataFrame,
                            col_datetime: str,
                            col_abertura: str,
                            col_maxima: str,
                            col_minima: str,
                            col_fechamento: str,
                            col_sombra: str = None,
                            titulo: str = 'Gráfico de Candles'):
    """
    Plota um gráfico de candles genérico com dados customizáveis.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        col_datetime (str): Nome da coluna de datas.
        col_abertura (str): Nome da coluna de abertura.
        col_maxima (str): Nome da coluna de máxima.
        col_minima (str): Nome da coluna de mínima.
        col_fechamento (str): Nome da coluna de fechamento.
        col_sombra (str, opcional): Nome da coluna para sobreposição como linha.
        titulo (str): Título do gráfico.
    """
    df = df.copy()
    df[col_datetime] = pd.to_datetime(df[col_datetime])

    fig = go.Figure()

    # Gráfico de candles
    fig.add_trace(go.Candlestick(
        x=df[col_datetime],
        open=df[col_abertura],
        high=df[col_maxima],
        low=df[col_minima],
        close=df[col_fechamento],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candles'
    ))

    # Linha de sombra opcional
    if col_sombra and col_sombra in df.columns:
        fig.add_trace(go.Scatter(
            x=df[col_datetime],
            y=df[col_sombra],
            mode='lines+markers',
            line=dict(color='white', width=1, dash='dot'),
            marker=dict(size=4, color='white'),
            name='Linha de Referência'
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title='Hora',
        yaxis_title='Preço',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.show()
