
# Experimento de LSTM com treinamento registrado e preparação dos dados reproduzida. Precisa revisar a avaliação antes de publicar.

-----


# Pipeline de Processamento de Dados de Ativos da B3

Este projeto realiza a extração, limpeza, transformação e carga incremental de dados financeiros de ativos da B3 (ex: PETR4, BBDC4, LREN3), estruturando-os em um banco de dados dimensional com modelo estrela.

O objetivo final é utilizar esses dados como base para desenvolvimento de modelos preditivos, especialmente modelos LSTM (Long Short-Term Memory), voltados à previsão de preços e comportamentos do mercado intradiário. A etapa de modelagem ainda está em desenvolvimento.

Este projeto é disponibilizado sob uma **licença personalizada de uso restrito**. Consulte a seção "Licença" ao final deste documento para mais detalhes.

---

## Ambiente de Execução

* O projeto foi concebido para execução em ambiente **Google Colab**, aproveitando a estrutura de diretórios padrão `/content/`.
* Certifique-se de montar corretamente a árvore de pastas (`Piloto_day_trade/`) dentro do ambiente Colab.
* Variáveis de ambiente sensíveis devem ser configuradas via `.env` (utilizando `python-dotenv`).

---

## Controle de Versão

O código-fonte está versionado utilizando **Git**. Recomenda-se clonar este repositório via GitHub para manter a rastreabilidade de alterações e contribuir com melhorias:

!git clone https://github.com/CarolBw/Piloto_day_trade.git


## Estrutura do Projeto

Piloto_day_trade/
├── data/
│   ├── raw/                  # Dados brutos extraídos (yfinance)
│   ├── cleaned/              # Dados limpos após filtragem de horários e duplicatas
│   ├── prepared/             # Reservado para features intermediárias
│   └── transformed/          # Dados finais prontos para carga
├── docs/                     # Documentações auxiliares
├── modelagem/
│   ├── catalog/              # Dicionário de dados e catálogos
│   ├── database/             # Estrutura e arquivos do banco
│   └── esquema/              # Modelo estrela e relações
├── models/                   # Modelos treinados ou salvos
├── notebooks/                # Análises exploratórias e estudos
├── reports/                  # Relatórios e saídas de modelos
│   └── figures/              # Gráficos gerados
├── src/
│   ├── modelagem_ml/         # Scripts de modelagem e previsão
│   ├── operacional/          # Utilitários operacionais
│   ├── pipeline/
│   │   ├── transformacao_dados.py  # Enriquecimento com indicadores e features
│   │   ├── carga_dados.py         # Carga incremental no SQLite
│   │   └── pipeline_<ticker>.py   # Pipeline ETL completo para um ticker
│   └── extrair_dados.py      # Extração e limpeza
├── tcc/                      # Documentos de TCC
├── README.md
├── requirements.txt
├── .env
├── .gitignore
└── LICENSE

## Etapas do Pipeline

### 1. Extração e Limpeza (`extrair_dados.py`)

* Baixa dados intradiários de ativos via `yfinance`.
* Converte datas e horários para timezone BRT.
* Renomeia colunas para nomes padronizados (`abertura`, `minimo`, `maximo`, `fechamento`, `volume`).
* Remove nulos, duplicatas e horários fora do pregão (09:55 - 18:05).
* Salva em `data/cleaned/`.

### 2. Transformação (`transformacao_dados.py`)

* Carrega os dados limpos e incrementa apenas registros ainda não processados.
* Ordena por data e hora.
* Calcula indicadores técnicos:

  * Médias móveis (SMA, EMA)
  * MACD e linha de sinal
  * RSI
  * OBV (On-Balance Volume)
  * Log-retorno e volatilidade
* Adiciona atributos temporais:

  * `dia_da_semana`, `hora_num`, `minuto`
* Adiciona estatísticas diárias (atuais e do dia anterior):

  * Fechamento, volume, máximo, mínimo
* Preenche nulos técnicos com a média da respectiva coluna.
* Salva em `data/transformed/`.

### 3. Carga Dimensional (`carga_dados.py`)

* Lê os dados transformados e conecta-se ao SQLite (`modelagem/database/`).
* Cria automaticamente as tabelas do modelo estrela, se não existirem:

  * **dim\_tempo**: data, hora, dia\_da\_semana
  * **dim\_indicadores**: indicadores técnicos e volume
  * **dim\_agregados**: horário numérico e estatísticas do dia anterior
  * **fato\_precos**: preços com referências às três dimensões
* Verifica se o par (`data`, `hora`) já foi carregado para evitar duplicações.
* Utiliza inserções condicionais (`INSERT OR IGNORE`).
* Garante integridade referencial entre fato e dimensões.
* Exibe a contagem de novos registros inseridos.

---

## Requisitos

* Python 3.10+
* Bibliotecas utilizadas no pipeline:

  * pandas, numpy, yfinance, matplotlib, seaborn
  * sqlite3, sqlalchemy
  * python-dotenv, gitpython

Instalação:

!pip install -r requirements.txt

---

## Execução

**Pipeline completo para um ativo**:

!python src/pipeline/pipeline_<ticker>.py


Substitua `<ticker>` por `BBDC4`, `LREN3`, etc. conforme desejado.

> **Importante:** Antes de executar o script principal do pipeline (ex: `pipeline_BBDC4.py`), verifique e ajuste os seguintes parâmetros no início do arquivo:
>
> * `ticker`: o código completo do ativo (ex: `BBDC4.SA`)
> * `intervalo`: intervalo de coleta dos dados (ex: `5m`, `15m`)
> * `dias`: número de dias de dados a serem buscados
> * Caminhos de entrada/saída dos dados e do banco SQLite, se necessário
>
> Esses parâmetros devem refletir suas necessidades específicas de coleta, transformação e armazenamento dos dados.

---

## Observações

* O pipeline é genérico e funciona para qualquer ticker da B3 aceito pelo Yahoo Finance (ex: `PETR4.SA`, `BBDC4.SA`, `LREN3.SA`).
* A transformação é incremental: evita retrabalho e reprocessamentos desnecessários.

---

## Em Desenvolvimento

* Criação do modelo preditivo LSTM para previsão de preços: estruturação, treinamento e avaliação
* Integração do pipeline com dashboard de visualização e monitoramento de dados e previsões
* Validações com testes automatizados
* Suporte a exportação em Parquet e bancos em nuvem

---

## 🔒 Licença

Copyright (c) 2025 Carolina Brescowitt

Todos os direitos reservados.

Este software é fornecido gratuitamente apenas para uso **pessoal, acadêmico e de pesquisa**.

O uso comercial deste software é estritamente proibido sem uma **licença comercial paga**, a ser negociada com o autor.

Empresas, startups, desenvolvedores ou qualquer entidade que deseje utilizar este código em produtos, serviços ou plataformas comerciais devem entrar em contato com o autor para **negociar os termos de licenciamento** (incluindo percentual, royalties ou valores fixos).

É proibida a redistribuição ou sublicenciamento sem autorização por escrito.

Para mais informações, entre em contato através do e-mail: [carolbrescowitt@yahoo.com.br](mailto:carolbrescowitt@yahoo.com.br)

---

Mantenha seus dados financeiros organizados e prontos para análise com este pipeline modular e extensível.
