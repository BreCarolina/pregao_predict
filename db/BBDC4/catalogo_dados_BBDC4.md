# Catálogo de Dados - BBDC4

Este documento descreve a estrutura e o conteúdo do banco de dados SQLite para o ticker **BBDC4**.

**Atualizado em:** 2025-06-23 16:03:22

---

## Tabela: `dim_agregados`

| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |
|---|---|---|---|---|
| `id_agregados` | `INTEGER` | Sim | Sim | Identificador único da dimensão de agregados diários. |
| `hora_num` | `INTEGER` | Não | Não | Componente da hora do dia (0-23). |
| `minuto` | `INTEGER` | Não | Não | Componente do minuto do dia (0-59). |
| `abertura_dia` | `REAL` | Não | Não | Preço de abertura do dia. |
| `minima_dia` | `REAL` | Não | Não | Preço mínimo do dia. |
| `maxima_dia` | `REAL` | Não | Não | Preço máximo do dia. |
| `fechamento_dia` | `REAL` | Não | Não | Preço de fechamento do dia. |
| `volume_dia` | `INTEGER` | Não | Não | Volume total de negociação do dia. |

## Tabela: `dim_indicadores`

| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |
|---|---|---|---|---|
| `id_indicadores` | `INTEGER` | Sim | Sim | Identificador único da dimensão de indicadores. |
| `RSI` | `REAL` | Não | Não | Índice de Força Relativa (Relative Strength Index). |
| `EMA_20` | `REAL` | Não | Não | Média Móvel Exponencial de 20 períodos. |
| `retorno` | `REAL` | Não | Não | Retorno percentual do candle atual. |

## Tabela: `dim_tempo`

| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |
|---|---|---|---|---|
| `id_tempo` | `INTEGER` | Sim | Sim | Chave estrangeira para a dimensão de tempo. |
| `data` | `TEXT` | Não | Não | Data do candle (YYYY-MM-DD). |
| `hora` | `TEXT` | Não | Não | Hora do candle (HH:MM:SS). |
| `datetime` | `TEXT` | Não | Não | Timestamp completo do candle (YYYY-MM-DD HH:MM:SS.ms). |
| `dia_da_semana` | `INTEGER` | Não | Não | Dia da semana (0=Segunda, 6=Domingo). |

## Tabela: `fato_precos`

| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |
|---|---|---|---|---|
| `id` | `INTEGER` | Sim | Sim | Chave primária da tabela de fatos. |
| `abertura` | `REAL` | Não | Não | Preço de abertura do candle. |
| `minima` | `REAL` | Não | Não | Preço mínimo do candle. |
| `maxima` | `REAL` | Não | Não | Preço máximo do candle. |
| `fechamento` | `REAL REAL` | Não | Não | Preço de fechamento do candle. |
| `volume` | `INTEGER` | Não | Não | Volume de negociação do candle. |
| `id_tempo` | `INTEGER` | Não | Não | Chave estrangeira para a dimensão de tempo. |
| `id_indicadores` | `INTEGER` | Não | Não | Identificador único da dimensão de indicadores. |
| `id_agregados` | `INTEGER` | Não | Não | Identificador único da dimensão de agregados diários. |

---

## Relacionamentos entre Tabelas

Este esquema utiliza um modelo estrela (Star Schema):

- **`fato_precos`** (Tabela de Fatos):
    - Contém as métricas de preço e volume por candle.
    - Relaciona-se com as tabelas de dimensão através das seguintes chaves estrangeiras:
        - `id_tempo` -> `dim_tempo.id_tempo`
        - `id_indicadores` -> `dim_indicadores.id_indicadores`
        - `id_agregados` -> `dim_agregados.id_agregados`

- **`dim_tempo`** (Dimensão de Tempo):
    - Descreve a granularidade temporal dos candles (data, hora, dia da semana).
    - Chave Primária: `id_tempo`

- **`dim_indicadores`** (Dimensão de Indicadores):
    - Armazena os valores dos indicadores técnicos (RSI, EMA_20, retorno).
    - Chave Primária: `id_indicadores`

- **`dim_agregados`** (Dimensão de Agregados Diários):
    - Contém os valores agregados diários (abertura do dia, mínima, máxima, fechamento, volume).
    - Chave Primária: `id_agregados`

---

## Observações Adicionais

- Os dados são armazenados na granularidade de candles de 5 minutos, conforme o intervalo de extração.
- As tabelas de dimensão são projetadas para reduzir a redundância de dados.
- Novos dados são inseridos incrementalmente, evitando duplicatas com base nas chaves de unicidade.
- Valores `NaN` para indicadores são tratados com interpolação linear, e um flag binário (`_indicador_faltante`) é adicionado para indicar sua ocorrência original.
- A coluna `is_outlier` indica a presença de valores extremos (outliers) no candle.
- `volume_candle` para candles preenchidos (lacunas) é definido como 0.
- Os agregados diários (`abertura_dia`, `minima_dia`, etc.) representam os valores do dia *daquele candle*.

