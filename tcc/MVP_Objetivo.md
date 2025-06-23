
### 1. Propósito do MVP

Este projeto tem como objetivo principal a criação de um pipeline para extração, transformação, carga, análise e previsão da movimentação intradiária dos preços de um ativo financeiro em intervalos de 5 minutos.
O modelo preditivo central será baseado em redes neurais recorrentes (LSTM), mas outras abordagens serão exploradas.
O MVP visa garantir previsões para embasar decisões estratégicas de day trade.

### 2. Problema a Ser Resolvido

A alta volatilidade dos mercados financeiros exige ferramentas robustas para antecipação de movimentos de preço.
A dificuldade está em capturar padrões de curto prazo e projetá-los com precisão.
Traders e investidores necessitam de um modelo que consiga interpretar os padrões históricos e transformá-los em previsões úteis.

### 3. Pipeline do Projeto

O projeto será estruturado em sete etapas:

1. **Extração e armazenamento dos dados brutos:**
Coleta de dados históricos do ativo em questao, utilizando a API do Yahoo Finance (yfinance).
Os dados serão armazenados em um repositório GitHub sincronizado com Google Colab, garantindo acesso remoto e backup na nuvem.

2. **Limpeza e organização dos dados:** Padronização de colunas, tratamento de dados ausentes, eliminação de duplicatas e organização cronológica.
Resultado salvo como `dados_limpos.csv`.

3. **Transformação e engenharia de features:** Adição de indicadores técnicos e features temporais.
Resultado salvo como `dados_transformados.csv`.

4. **Modelagem e estruturação do banco de dados:**

   a) Organização em arquivos CSV:

   - `dados_brutos.csv`: dados originais extraídos da API.
   - `dados_limpos.csv`: após limpeza e padronização.
   - `dados_transformados.csv`: após adição de features técnicas.
   - `dados_preparados.csv`: versão padronizada e normalizada dos dados dividida em treino teste e validação.

   b) Banco de dados dimensional:
      Modelagem de dados no esquema estrela para facilitar a análise de fatos.

   - **Fato**: `fato_precos`, contendo os valores de fechamento e chaves para dimensões.
   - **Dimensões**:
     - `dim_tempo`: atributos temporais.
     - `dim_indicadores`: indicadores técnicos.
     - `dim_agregados`: dados agregados para enriquecer a análise.
     - `dim_operacional': dados operacionais.

   c) Catalogo de dados

  - Um **Catálogo de Dados** será elaborado com descrição, domínio, categorias e linhagem de cada variável.

5. **Carga e Pipeline ETL:** Pipeline estruturado com as seguintes etapas:

   - **Extração:** via API do Yahoo Finance.
   - **Limpeza:** tratamento e estruturação básica.
   - **Transformação:** geração de variáveis técnicas e derivadas.
   - **Carga:** integração dos dados transformados no banco de dados dimensional.

6. **Treinamento e ajuste do modelo:** Implementação e ajuste do modelo preditivo com base LSTM, com avaliação por métricas como MSE e R².

7. **Interpretação dos resultados e resposta às perguntas:** Validação das previsões.

### 4. Perguntas a Serem Respondidas

01 - Qual horário do dia é comum maior alta do ativo?
02 - Existe algum padrão proximo ao horário de fechamento do pregão?
03 - Qual progressão média do ativo por mes?
04 - Com qual volume de negociações normalmente temos uma inversão?
05 - Qual correlação do RSI com a inversão?
06 - Qual correlação do RSI com o preço de fechamento?
07 - É possivel gerar a previsão completa do pregão de 5 em 5 minutos com precisão?
08 - É possivel prever as targets globais do ativo no pregao? (Abertura, fechamento, minima e máxima)
09 - Qual tendencia de preço do ativo em um ano?
10 - Existe algum padrao que constitui a queda do preço do ativo?
11 - Os dados do dia anterior fornecem informações suficientes para a previsão do dia seguinte?
12 - A modelagem com LSTM captura corretamente as tendências de curto prazo?
13 - Quais indicadores técnicos e features são mais relevantes para melhorar a acurácia do modelo?
14 - Como considerar corretamente as quebras de fim de semana?
15 - A normalização e padronização das variáveis melhora a precisão do modelo?

### 5. Critérios de Sucesso

Para que o MVP seja considerado bem-sucedido o esperado é que:

1. O pipeline de extração, transformação e previsão funcione de forma eficiente.
2. O modelo consiga prever a movimentação dos preços com um erro médio aceitável.
3. A previsão de targets globais (abertura, máxima, mínima, fechamento) seja consistente com os valores reais em pelo menos 90%.
4. O modelo consiga lidar corretamente com fins de semana e feriados.
5. As previsões sejam suficientes para auxiliar na tomada de decisão de trading.

