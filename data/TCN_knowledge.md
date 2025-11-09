# Base de Conhecimento: Darts TCNModel

Este documento serve como uma base de conhecimento técnica e um guia de implementação para o `TCNModel` (Temporal Convolutional Network) dentro do pacote Darts. Ele é destinado a um agente de programação, detalhando a arquitetura, o pré-processamento de dados, a parametrização e as limitações operacionais para permitir o desenvolvimento de uma solução de forecasting robusta.

## I. Arquitetura Fundamental e Teoria Operacional da TCN

A implementação do `TCNModel` no Darts é baseada nos princípios estabelecidos no paper "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling".¹ Compreender esta arquitetura é fundamental para a parametrização correta.

### 1.1. Propósito e Vantagens Principais (TCN vs. RNN)

O paper de referência posiciona a TCN como uma alternativa superior às arquiteturas recorrentes (RNNs), como LSTMs e GRUs, para modelagem de sequências.¹ A avaliação empírica demonstrou que as TCNs superam as RNNs em uma ampla gama de tarefas, exibindo uma memória efetiva "substancialmente mais longa" na prática.¹

As vantagens arquitetônicas centrais são ¹:

-   **Paralelismo**: As convoluções, ao contrário das etapas sequenciais de uma RNN, podem ser processadas em paralelo, resultando em treinamento e inferência mais rápidos.
-   **Tamanho Flexível do Campo Receptivo**: O "histórico" ou "memória" do modelo (Campo Receptivo) pode ser controlado explicitamente ajustando hiperparâmetros como profundidade da rede, tamanho do kernel e dilatação.
-   **Gradientes Estáveis**: O caminho da retropropagação em uma TCN não segue a direção temporal da sequência. Isso evita os problemas de gradiente explosivo/desvanescente (exploding/vanishing gradients) que são notórios em RNNs.
-   **Baixo Requisito de Memória (Treinamento)**: Os filtros convolucionais são compartilhados através da camada, tornando o TCN mais eficiente em termos de memória durante o treinamento em comparação com as múltiplas portas de LSTMs e GRUs.

### 1.2. O Pilar 1: Convoluções Causais (Causal Convolutions)

A característica que define a TCN para forecasting é a convolução causal.

-   **Conceito**: Uma convolução causal garante que a previsão para o tempo $t$ (a saída $y_t$) dependa apenas de entradas do tempo $t$ e anteriores ($x_0,..., x_t$).¹ Isso impõe a restrição fundamental de que não pode haver "vazamento" de informações futuras para o passado.
-   **Implementação**: Isso não é um tipo de camada especial, mas sim a aplicação de padding assimétrico (preenchimento assimétrico). O padding de zeros é aplicado apenas no lado esquerdo (o "passado") da sequência de entrada antes da operação `Conv1d`.² O código-fonte do Darts confirma isso calculando um `left_padding` e aplicando-o com `F.pad(x, (left_padding, 0))`.³

### 1.3. O Pilar 2: Convoluções Dilatadas (Dilated Convolutions)

O Pilar 1 (convoluções causais) por si só é ineficiente. Em uma rede convolucional padrão, o campo receptivo (RF) cresce linearmente com a profundidade da rede. Para obter um histórico longo, a rede precisaria ser "extremamente profunda ou ter filtros muito grandes".¹

-   **Solução**: A TCN emprega convoluções dilatadas. Uma dilatação introduz "buracos" fixos em um filtro, pulando entradas com um passo definido (o fator de dilatação $d$).¹
-   **Crescimento Exponencial**: A arquitetura TCN empilha camadas e aumenta o fator de dilatação exponencialmente com a profundidade (nível $i$ da rede). A dilatação é calculada como $d = \text{dilation\_base}^i$.³ Com um `dilation_base` de 2, as dilatações seriam 1, 2, 4, 8, 16, e assim por diante. Isso permite que o campo receptivo (RF) cresça exponencialmente com a profundidade da rede, permitindo que o modelo alcance "cobertura de histórico completa" com um número relativamente pequeno de camadas.¹

### 1.4. O Pilar 3: Blocos Residuais (Residual Blocks)

O Pilar 2 (convoluções dilatadas) resolve o problema do RF, mas, para ser eficaz, requer uma rede profunda (por exemplo, 8, 10 ou 12 camadas).¹ Redes muito profundas sofrem de degradação de gradiente (dificuldade em treinar).

-   **Solução**: A TCN substitui camadas convolucionais simples por Blocos Residuais.¹ Um bloco residual implementa uma *skip connection* (conexão de atalho) que adiciona a entrada do bloco ($x$) à saída das transformações do bloco ($F(x)$).¹ A operação é $o = \text{Activation}(x + F(x))$.
-   **Manipulação de Dimensão**: Em redes convolucionais, a entrada $x$ e a saída $F(x)$ podem ter larguras de canal (número de filtros) diferentes. Se as dimensões não corresponderem, uma convolução 1x1 é aplicada ao shortcut ($x$) para garantir que os tensores tenham a mesma forma antes da adição.¹ O código-fonte do Darts implementa isso explicitamente.³

Esses três pilares formam uma cadeia de solução de problemas:

1.  A **Convolução Causal** define o problema (previsão sem vazamento), mas é ineficiente (RF linear).
2.  A **Convolução Dilatada** resolve o problema de RF (torna-o exponencial), mas cria um novo problema (necessidade de redes profundas).
3.  O **Bloco Residual** resolve o problema de profundidade, estabilizando o treinamento para redes profundas.

## II. Preparação e Pré-processamento de Dados para TCNModel

O `TCNModel` do Darts exige um fluxo de trabalho de preparação de dados rigoroso, centrado no objeto `TimeSeries`.

### 2.1. O Objeto `TimeSeries`: O Contrato de Entrada

A entrada fundamental para todos os modelos Darts é o objeto `darts.TimeSeries`.⁴ O primeiro passo de qualquer pipeline de pré-processamento deve ser converter dados brutos (ex: DataFrames do Pandas) para este formato.

-   **Implementação**: `ts = TimeSeries.from_dataframe(df, "col_tempo", ["col_valor"])`.⁴
-   **Valores Ausentes**: O Darts fornece utilitários para lidar com valores ausentes, o que deve ser feito após a criação do objeto: `ts_filled = fill_missing_values(ts, "auto")`.⁵

### 2.2. Divisão Cronológica de Dados (Treino/Validação)

A divisão de séries temporais para backtesting deve ser cronológica para evitar o vazamento de dados futuros. O uso de divisões aleatórias (como `train_test_split` do scikit-learn) é um erro metodológico grave.

-   **Implementação**: O Darts facilita isso com o método `.split_after()`. `train, val = ts.split_after(pd.Timestamp("19580801"))`.⁵

### 2.3. Normalização e Escalonamento de Dados: Prevenção de Vazamento Crítico

Modelos de Deep Learning são sensíveis à escala dos dados de entrada. O escalonamento é obrigatório, e deve ser feito de forma a prevenir o vazamento de informações do conjunto de validação/teste para o conjunto de treinamento.

O `darts.dataprocessing.Scaler` é projetado para este fim. O fluxo de trabalho correto é ⁵:

1.  **Instanciar o Scaler**: `scaler = Scaler()`
2.  **Ajustar (Fit) e Transformar apenas nos dados de treino**: `train_scaled = scaler.fit_transform(train)`
    -   Esta etapa calcula as estatísticas (ex: $\mu$, $\sigma$) apenas do conjunto `train` e as armazena no objeto `scaler`.
3.  **Transformar (Transform) os conjuntos de validação e teste**:
    -   `val_scaled = scaler.transform(val)`
    -   `test_scaled = scaler.transform(test)`
    -   Estas etapas aplicam a transformação usando as estatísticas salvas do conjunto de treino, evitando qualquer vazamento.

Covariáveis que são *one-hot encoded* (como atributos de mês ou dia) não requerem escalonamento.⁵

## III. Gerenciamento Técnico de Covariáveis (`past_covariates`)

O gerenciamento de covariáveis (dados externos) é uma fonte comum de erros. O `TCNModel` no Darts tem limitações específicas.

### 3.1. Limitação Fundamental: Apenas `past_covariates`

O `TCNModel` do Darts, conforme documentado, suporta apenas `past_covariates` (covariáveis passadas).⁶

-   **Não suporta `future_covariates`**: (Covariáveis conhecidas no futuro, como feriados ou promoções planejadas).
-   **Não suporta `static_covariates`**: (Covariáveis que não mudam no tempo, como IDs de loja ou SKUs de produto).

Esta é uma limitação severa. Se `future_covariates` ou `static_covariates` forem essenciais para o problema, um modelo diferente (como `TFTModel`) deve ser selecionado.

### 3.2. Criação e Formato de `past_covariates`

As covariáveis passadas devem ser formatadas como objetos `TimeSeries`.⁶ Uma fonte comum são atributos de data/hora, que podem ser gerados usando utilitários do Darts:

-   **Implementação**: `month_series = datetime_attribute_timeseries(ts, attribute="month", one_hot=True)`.⁵
-   **Dimensionalidade**: Para atributos categóricos como "mês", `one_hot=True` é essencial. Isso cria uma `TimeSeries` com 12 colunas (dimensões).⁵ A dimensionalidade total da entrada para o modelo será `target_series.width + past_covariates.width` (ex: $1 + 12 = 13$).

### 3.3. O "Fatiamento Inteligente" e Requisitos de Alinhamento Temporal

O Darts "fatiá-los-á de forma inteligente".⁶ Durante o `fit()` ou `predict()`, pode-se passar a `TimeSeries` de covariável completa (ex: cobrindo treino e validação). O Darts alinhará automaticamente a covariável à série alvo usando seus índices de tempo e extrairá o chunk temporal correto.⁵

-   **Requisito Crítico**: Este fatiamento automático só funciona se a `TimeSeries` da série alvo e a `TimeSeries` das covariáveis passadas tiverem índices de tempo perfeitamente alinhados e com a mesma frequência. Um desalinhamento de um passo de tempo ou uma frequência diferente (ex: diário vs. horário) causará falha silenciosa, com o modelo recebendo dados de covariável incorretos.

### 3.4. Requisitos de Covariáveis para Treinamento (`fit`)

Para treinar um modelo com covariáveis:

-   `past_covariates` deve cobrir pelo menos o mesmo período de tempo da `series` (treinamento).⁶
-   Se `val_series` for fornecida (altamente recomendado), `val_past_covariates` também deve ser fornecido.⁵

-   **Implementação**:
    ```python
    model.fit(series=train_scaled,
              past_covariates=covariates_scaled,
              val_series=val_scaled,
              val_past_covariates=covariates_scaled) # Passando a série de covariáveis completa
    ```

### 3.5. Requisitos de Covariáveis para Previsão (`predict`)

Os requisitos de covariáveis no momento da previsão dependem da estratégia de previsão (determinada por `n` vs. `output_chunk_length`, ver Seção VI) ⁶:

-   **Caso 1: Previsão One-Shot ($n \le \text{output\_chunk\_length}$)**:
    -   A `past_covariates` deve cobrir pelo menos o mesmo período de tempo da `series` de entrada (ou seja, os últimos `input_chunk_length` pontos antes da previsão).
-   **Caso 2: Previsão Auto-Regressiva ($n > \text{output\_chunk\_length}$)**:
    -   A `past_covariates` deve cobrir o período da `series` de entrada MAIS os próximos $n - \text{output\_chunk\_length}$ passos de tempo após o final da `series`.

O "Caso 2" (auto-regressivo) implica que as `past_covariates` devem ser conhecidas no futuro. Isso só é viável para covariáveis "determinísticas" (ex: atributos de data/hora, como `month_series` ⁵). Se a `past_covariates` for uma variável exógena que também precisa ser prevista (ex: "preço do petróleo"), o modo auto-regressivo não pode ser usado.

## IV. Dicionário de Parâmetros de Implementação (`TCNModel.__init__`)

A instanciação correta do `TCNModel` é crucial. A tabela a seguir detalha os parâmetros `__init__` essenciais, com base na documentação do código-fonte.³

| Parâmetro | Padrão | Descrição Técnica e Finalidade |
| :--- | :--- | :--- |
| `input_chunk_length` | (Obrigatório) | **O que é**: O número de passos de tempo (comprimento da sequência) que o modelo recebe como entrada em cada amostragem durante o treinamento. <br> **Finalidade**: Define o "pedaço" de dados que o modelo vê. <br> **Instrução**: Deve ser definido como pelo menos o tamanho do Campo Receptivo (RF) desejado (ver 4.1). Se `input_chunk_length < RF`, o modelo não pode usar seu histórico completo. |
| `output_chunk_length` | (Obrigatório) | **O que é**: O número de passos de tempo que o modelo prevê de uma só vez (one-shot) em uma única passagem para frente. <br> **Finalidade**: Controla a estratégia de previsão. <br> **Instrução**: Parâmetro crítico. Deve ser definido como igual ao horizonte de previsão `n` mais longo esperado para forçar a previsão "One-Shot" (ver Seção VI).⁵ |
| `kernel_size` | 3 | **O que é**: O tamanho do filtro (kernel) em cada camada `nn.Conv1d`.³ <br> **Finalidade**: Controla o campo receptivo local. <br> **Instrução**: O paper ¹ sugere que valores maiores (ex: 7) podem ser benéficos, mas `k=3` é um padrão comum. |
| `num_filters` | 3 | **O que é**: O número de canais de saída (filtros) em cada camada convolucional intermediária.³ <br> **Finalidade**: Controla a capacidade ou largura do modelo. Análogo ao `hidden_size` em uma RNN. <br> **Instrução**: O padrão `num_filters=3` ³ é extremamente pequeno e inadequado para problemas reais. Valores significativamente maiores (ex: 32, 64, 128) são necessários. |
| `num_layers` | None | **O que é**: O número de `_ResidualBlock` a serem empilhados.³ <br> **Finalidade**: O controlador primário do Campo Receptivo (RF). O RF cresce exponencialmente com `num_layers`. <br> **Instrução**: Deve ser definido explicitamente para controlar a memória do modelo. |
| `dilation_base` | 2 | **O que é**: A base do expoente para a dilatação ($d = \text{dilation\_base}^i$).³ <br> **Finalidade**: Controla a taxa de crescimento do RF. <br> **Instrução**: `dilation_base=2` é o padrão do paper ¹ e quase sempre a escolha correta. |
| `weight_norm` | False | **O que é**: Booleano para aplicar Normalização de Peso (`nn.utils.parametrizations.weight_norm`) aos filtros `Conv1d`.³ <br> **Finalidade**: Ajuda a estabilizar e acelerar o treinamento.¹ <br> **Instrução**: Recomendado definir como `True` para replicação mais próxima do paper.¹ |
| `dropout` | 0.2 | **O que é**: A taxa de dropout (desativação) aplicada após cada camada convolucional dentro do bloco residual.³ <br> **Finalidade**: Regularização para prevenir overfitting. <br> **Instrução**: Darts usa `MonteCarloDropout` ³, permitindo previsões probabilísticas em tempo de inferência.⁹ |
| `output_chunk_shift` | 0 | **O que é**: O número de passos para deslocar o chunk de saída no futuro.³ <br> **Finalidade**: Cria uma "lacuna" entre o fim da entrada e o início da saída. <br> **Instrução**: Raramente usado. Manter o padrão 0. |

### 4.1. Cálculo do Campo Receptivo (Receptive Field - RF)

O `input_chunk_length` é um parâmetro de fatiamento. O histórico real que o modelo utiliza é o Campo Receptivo (RF), determinado por `kernel_size`, `num_layers` e `dilation_base`.

A implementação `_ResidualBlock` do Darts usa duas camadas `Conv1d` por bloco, ambas compartilhando a mesma dilatação.³ A fórmula para o RF (em passos de tempo) é:

$RF = 1 + 2 \times (\text{kernel\_size} - 1) \times \sum_{i=0}^{\text{num\_layers} - 1} (\text{dilation\_base}^i)$

**Instrução Crítica**: O agente deve calcular o RF esperado e garantir que `input_chunk_length >= RF`. Se o RF calculado for 500, um `input_chunk_length=100` subutilizará drasticamente o modelo, pois a rede não terá dados suficientes para preencher seus filtros mais profundos.

## V. Arquitetura Interna: Decodificando o `_ResidualBlock` do Darts

Uma análise do código-fonte `_ResidualBlock` ³ confirma exatamente como a teoria (Seção I) é implementada no Darts.

### 5.1. Análise do Código-Fonte `_ResidualBlock`

-   **Implementação da Dilatação**: O bloco `i` (indexado por `nr_blocks_below`) define sua dilatação como:
    ```python
    self.conv1 = nn.Conv1d(..., dilation=(dilation_base**nr_blocks_below))
    self.conv2 = nn.Conv1d(..., dilation=(dilation_base**nr_blocks_below))
    ```
    -   **Confirmação**: A dilatação é $d = \text{dilation\_base}^i$, implementando o crescimento exponencial do RF.

-   **Implementação da Convolução Causal**: O padding é calculado e aplicado antes de cada convolução:
    ```python
    left_padding = (self.dilation_base**self.nr_blocks_below) * (self.kernel_size - 1)
    x = F.pad(x, (left_padding, 0))
    ```
    -   **Confirmação**: O padding é aplicado apenas à esquerda (`(left_padding, 0)`), garantindo causalidade estrita.

-   **Implementação da Conexão Residual (com Shortcut 1x1)**:
    ```python
    # Na inicialização (__init__):
    if input_dim != output_dim:
        self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    # Na passagem para frente (forward):
    residual = x
    #... (convoluções x = self.conv2(...))
    if self.conv1.in_channels != self.conv2.out_channels:
        residual = self.conv3(residual)
    x = x + residual
    ```
    -   **Confirmação**: Implementa o Pilar 3. O shortcut de convolução 1x1 (`self.conv3`) só é criado e usado se as dimensões de canal da entrada e saída do bloco não corresponderem.

### 5.2. O Fluxo de Dimensão (Encoder-Processador-Decoder)

A lógica `input_dim` vs. `output_dim` no `_ResidualBlock` ³ revela uma estrutura interna de "Encoder-Processador-Decoder" baseada no `num_filters`:

-   **Bloco 0 (Encoder)**: `input_dim = input_size` (ex: 13, vindo de 1 alvo + 12 covs). `output_dim = num_filters` (ex: 64). O `self.conv3` é usado para mapear $13 \rightarrow 64$.
-   **Blocos 1 a N-2 (Processador)**: `input_dim = num_filters` (64). `output_dim = num_filters` (64). O `self.conv3` não é usado (a conexão residual é uma identidade pura).
-   **Bloco N-1 (Decoder)**: `input_dim = num_filters` (64). `output_dim = target_size` (ex: 1). O `self.conv3` é usado para mapear $64 \rightarrow 1$.

Isso confirma que `num_filters` pode ser definido independentemente das dimensões de entrada/saída, pois o modelo lida automaticamente com o mapeamento para e do espaço de processamento interno.

## VI. Guia de Implementação: Treinamento (`fit`) e Previsão (`predict`)

Esta seção fornece as instruções táticas finais para o agente.

### 6.1. O Processo de Treinamento: `model.fit()`

O método `fit()` treina o modelo. Para um treinamento robusto, é essencial usar um conjunto de validação para monitoramento e callbacks (como `EarlyStopping`).

-   **Parâmetros Essenciais**: ⁵
    -   `series`: A `TimeSeries` de treinamento (escalada).
    -   `past_covariates`: A `TimeSeries` de covariáveis (escalada, se aplicável).
    -   `val_series`: A `TimeSeries` de validação (escalada).
    -   `val_past_covariates`: As covariáveis para o conjunto de validação.

### 6.2. O Processo de Previsão: `model.predict()` e o Parâmetro `n`

O método `predict()` gera previsões. O parâmetro chave é `n`, o horizonte de previsão desejado (quantos passos no futuro prever).⁸

É crucial entender que `n` (parâmetro de inferência) não é o mesmo que `output_chunk_length` (parâmetro de arquitetura `__init__`).⁸

### 6.3. A Relação Crítica: `n` vs. `output_chunk_length`

A relação entre `n` e `output_chunk_length` dita a estratégia de previsão.

-   **Caso 1 (Preferencial): Previsão One-Shot**
    -   **Condição**: $n \le \text{output\_chunk\_length}$.⁵
    -   **Exemplo**: `TCNModel(..., output_chunk_length=12)`, `model.predict(n=6)`.⁵
    -   **Como Funciona**: O modelo executa uma passagem para frente, gerando 12 pontos de previsão. O Darts então corta e retorna apenas os primeiros $n=6$ pontos.
    -   **Vantagens**: Rápido (uma passagem), estável (sem acúmulo de erro) e requisitos de covariáveis mais simples (Caso 1 em 3.5).

-   **Caso 2 (A Evitar): Previsão Auto-Regressiva**
    -   **Condição**: $n > \text{output\_chunk\_length}$.⁶
    -   **Exemplo**: `TCNModel(..., output_chunk_length=12)`, `model.predict(n=36)`.
    -   **Como Funciona**:
        1.  O modelo prevê os passos 1-12.
        2.  Ele alimenta suas próprias previsões (passos 1-12) de volta como entrada.
        3.  Ele prevê os passos 13-24.
        4.  Ele repete o processo 3 vezes até que $n=36$ pontos sejam gerados.
    -   **Desvantagens**: Lento (múltiplas passagens) e instável (erros nas previsões 1-12 são amplificados nas previsões 13-24).
    -   **Desvantagem Crítica de Covariável**: Conforme ⁶, este modo requer que `past_covariates` sejam conhecidas $n - \text{output\_chunk\_length}$ passos no futuro (Caso 2 em 3.5), o que é frequentemente impossível.

**PRESCRiÇÃO (Instrução Direta para o Agente):**

O agente deve sempre instanciar o `TCNModel` com `output_chunk_length` igual ao `n` (horizonte de previsão) máximo que o usuário planeja solicitar. Se o usuário solicitar `model.predict(n=36)`, o agente deve ter inicializado o modelo com `output_chunk_length=36` (ou mais). Isso força o modelo a sempre usar a estratégia One-Shot (Caso 1), que é mais rápida, estável e tem requisitos de covariáveis mais simples.

## VII. Limitações Conhecidas e Considerações Práticas

### 7.1. Limitações da Arquitetura TCN (Baseada no Paper)

O paper TCN lista duas desvantagens principais ¹:

-   **Armazenamento de Dados Durante a Avaliação**: Ao contrário de uma RNN (que só precisa de seu último estado oculto $h_t$), o TCN precisa de um buffer de entrada bruto de tamanho `input_chunk_length` para fazer cada nova previsão. Em um backtest (`historical_forecasts`), isso significa que o TCN deve re-executar convoluções em chunks longos a cada passo, o que pode tornar a inferência sequencial mais lenta que a de uma RNN otimizada.
-   **Mudança de Parâmetro para Transferência de Domínio**: O Campo Receptivo (RF) é fixo pelos hiperparâmetros `num_layers`, `kernel_size` e `dilation_base`. Um modelo treinado em dados diários (onde um RF de 100 significa 100 dias) falhará se for transferido para dados horários (onde 100 passos são apenas ~4 dias) sem ser re-parametrizado (ex: aumentando `num_layers`) e re-treinado.

### 7.2. Limitações Específicas da Implementação Darts (`TCNModel`)

-   **Sem Suporte a `future_covariates`**: Conforme ⁶, o modelo não pode usar nativamente informações futuras conhecidas, como feriados ou promoções.
-   **Sem Suporte a `static_covariates`**: Conforme ⁶, o modelo não pode usar atributos estáticos (IDs de loja, SKUs). Isso torna o `TCNModel` do Darts uma escolha inadequada para treinar um único modelo global em milhares de séries temporais heterogêneas, onde IDs estáticos são essenciais para diferenciar as séries.
-   **Ajuste de Parâmetros (Tuning)**: O desempenho do TCN é altamente sensível ao RF (controlado por `num_layers`, `kernel_size`, `dilation_base`) e à capacidade do modelo (`num_filters`). Os padrões, especialmente `num_filters=3` ³, são inadequados para tarefas reais.² Um processo de *hyperparameter optimization* é necessário para encontrar uma arquitetura viável.