# Catálogo de Dados - LREN3

**Atualizado em:** 2025-07-02 02:59:25

---

## Tabela: `dados_limpos`

| Nome da Coluna | Tipo de Dado (SQLite) | Nulo? | Chave Primária? | Descrição |
|---|---|---|---|---|
| `id` | `INTEGER` | Sim | Sim | Chave primária da tabela. |
| `ativo` | `TEXT` | Não | Não | Código do ativo negociado. |
| `data` | `TEXT` | Não | Não | Data da negociação (YYYY-MM-DD). |
| `hora` | `TEXT` | Não | Não | Hora da negociação (HH:MM:SS). |
| `abertura` | `REAL` | Não | Não | Preço de abertura do candle. |
| `maximo` | `REAL` | Não | Não | Preço máximo do candle. |
| `minimo` | `REAL` | Não | Não | Preço mínimo do candle. |
| `fechamento` | `REAL` | Não | Não | Preço de fechamento do candle. |
| `volume` | `REAL` | Não | Não | Volume negociado no candle. |
| `quantidade` | `REAL` | Não | Não | Quantidade de negócios no candle. |

---

## Observações

- A base contém apenas a tabela `dados_limpos`, com dados de candles de 5 minutos.
- Cada linha representa um momento único da negociação (ativo + data + hora).
- A estrutura é simples e direta, ideal para análises e modelagens temporais.
