
import pandas as pd
import plotly.graph_objects as go

# Dados pivotados
data = {
    'Data': ['2025-05-09 14:00', '2025-05-09 14:02', '2025-05-09 14:04', '2025-05-09 14:06'],
    'Abertura': [138100, 138200, 138150, 138250],
    'Máxima': [138300, 138250, 138280, 138350],
    'Mínima': [138050, 138100, 138120, 138200],
    'Fechamento': [138200, 138150, 138250, 138300],
    'Volume': [35000000, 27000000, 42000000, 31000000]
}

df = pd.DataFrame(data)
df['Data'] = pd.to_datetime(df['Data'])

# Criar gráfico de candles
fig = go.Figure(data=[
    go.Candlestick(
        x=df['Data'],
        open=df['Abertura'],
        high=df['Máxima'],
        low=df['Mínima'],
        close=df['Fechamento'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )
])

# Adicionar gráfico de volume
fig.add_trace(go.Bar(
    x=df['Data'],
    y=df['Volume'],
    marker_color='rgba(0,150,0,0.3)',
    name='Volume',
    yaxis='y2'
))

# Layout com dois eixos Y
fig.update_layout(
    title='Gráfico de Candles (Ibovespa Mini)',
    xaxis_title='Hora',
    yaxis_title='Preço',
    yaxis2=dict(
        title='Volume',
        overlaying='y',
        side='right',
        showgrid=False
    ),
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

fig.show()
