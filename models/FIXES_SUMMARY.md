# Resumo das Correções - TCN_new.ipynb

## Data: 2025-11-10

### Problemas Identificados

1. **Seção 5 (Normalização de Dados)**:
   - Erro de linter: `'Não é possível acessar o atributo "values" para a classe "list[TimeSeries]"'`
   - Valores NaN para estatísticas (Média, Desvio Padrão, Mínimo, Máximo)

2. **Seção 7 (Treinamento do Modelo)**:
   - `train_loss` e `val_loss` iguais a `nan.0` durante o treinamento

3. **Seção 8 (Avaliação)**:
   - Nenhuma previsão gerada

### Correções Implementadas

#### 1. Seção 5 - Normalização de Dados (linhas 343-358)

**Problema**: O código tentava acessar `.values()` em um objeto `TimeSeries`, mas o método correto para extrair dados de forma robusta é usar `.pd_dataframe()`.

**Antes**:
```python
train_values = train_scaled.values().flatten()
```

**Depois**:
```python
# TimeSeries.values() retorna um numpy array 2D, acessar com pd_dataframe() é mais robusto
train_df = train_scaled.pd_dataframe()
train_values = train_df.values.flatten()
```

**Justificativa**: 
- O método `.pd_dataframe()` retorna um DataFrame pandas que é mais fácil de manipular
- Evita problemas com a estrutura multidimensional retornada por `.values()`
- Garante compatibilidade com diferentes versões do Darts

#### 2. Seção 8 - Avaliação no Conjunto de Validação (linhas 488-501)

**Problema**: O `historical_forecasts` não conseguia fazer previsões porque `val_scaled` é um objeto TimeSeries independente, sem acesso aos dados de treinamento que vêm antes dele temporalmente.

**Conceito Importante**: Embora o conjunto de treinamento tenha muito mais do que 168 horas de dados (na verdade, tem cerca de 18.000+ horas, pois são 70% de dados de 2022-2025), o método `historical_forecasts` só tem acesso ao objeto TimeSeries passado no parâmetro `series`. Como `train_scaled` e `val_scaled` são objetos **separados**, quando passamos apenas `val_scaled`, o método não consegue "olhar para trás" para os dados de treino.

**Antes**:
```python
val_predictions = model.historical_forecasts(
    series=val_scaled,  # Apenas validação - sem contexto histórico!
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Depois**:
```python
# Concatenar train e val para que historical_forecasts tenha acesso ao histórico
train_val_scaled = train_scaled.append(val_scaled)

# Fazer previsões começando no início do período de validação
val_predictions = model.historical_forecasts(
    series=train_val_scaled,  # Agora tem train + val
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    start=len(train_scaled),  # Começar previsões no índice onde val começa
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Justificativa**:
- O parâmetro `start=len(train_scaled)` diz ao método: "comece a fazer previsões a partir deste índice"
- Isso significa que ele faz previsões apenas no período de validação, mas tem acesso aos dados de treino para contexto
- É análogo a dizer: "use os dados de treino como histórico, mas faça previsões apenas na validação"
- **Não estamos treinando com dados de validação** - apenas fornecendo o contexto histórico necessário para fazer previsões

#### 3. Seção 9 - Avaliação no Conjunto de Teste (linhas 544-556)

**Problema**: Mesma questão da seção 8 - `test_scaled` é um objeto independente sem acesso aos dados anteriores.

**Antes**:
```python
test_predictions = model.historical_forecasts(
    series=test_scaled,  # Apenas teste - sem contexto!
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Depois**:
```python
# Concatenar train+val+test para fornecer todo o histórico
full_scaled = train_scaled.append(val_scaled).append(test_scaled)

# Fazer previsões começando no início do período de teste
test_predictions = model.historical_forecasts(
    series=full_scaled,  # Tem train + val + test
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    start=len(train_scaled) + len(val_scaled),  # Começar no índice onde test começa
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Justificativa**:
- Similar à seção 8
- `start=len(train_scaled) + len(val_scaled)` indica que as previsões devem começar no início do teste
- O modelo usa train+val como contexto histórico, mas faz previsões apenas no teste
- Garante avaliação realista: o modelo vê o histórico completo disponível até o ponto de previsão

### Impacto Esperado

Com estas correções:

1. ✅ **Seção 5**: As estatísticas do conjunto normalizado devem exibir valores corretos (média ≈ 0, desvio padrão ≈ 1)
2. ✅ **Seção 7**: O treinamento deve completar com valores numéricos válidos para `train_loss` e `val_loss`
3. ✅ **Seção 8**: Previsões devem ser geradas com sucesso no conjunto de validação
4. ✅ **Seção 9**: Previsões devem ser geradas com sucesso no conjunto de teste

### Observações Técnicas

#### Por que concatenar se já temos 70% dos dados (18.000+ horas)?

**Resposta Curta**: Porque `train_scaled`, `val_scaled` e `test_scaled` são objetos TimeSeries **independentes**. O método `historical_forecasts` só vê o que você passa no parâmetro `series`.

**Explicação Detalhada**:

1. **Os dados de treino têm muitos dados**: Sim! 70% de dados de 2022-2025 (~3 anos) = aproximadamente 18.000-20.000 horas. Isso é muito mais do que os 168 horas necessários para `input_chunk_length`.

2. **Mas os objetos são separados**: Quando fazemos:
   ```python
   train = ts[:train_size]
   val = ts[train_size:train_size + val_size]
   ```
   Criamos dois objetos TimeSeries diferentes. `val` não "sabe" sobre `train`.

3. **Como `historical_forecasts` funciona**:
   - Ele recebe um TimeSeries e faz previsões em vários pontos ao longo dele
   - Para fazer uma previsão no tempo `t`, ele precisa olhar para trás `input_chunk_length` passos
   - Se você passar apenas `val_scaled`, quando ele tenta fazer a PRIMEIRA previsão (no início de val), ele tenta olhar 168 horas para trás, mas essas 168 horas estão em `train_scaled`, que não foi passado!

4. **A concatenação resolve isso**:
   ```python
   train_val_scaled = train_scaled.append(val_scaled)
   ```
   Agora `historical_forecasts` vê um único TimeSeries contínuo e pode acessar o histórico completo.

5. **O parâmetro `start` garante que só fazemos previsões na validação**:
   ```python
   start=len(train_scaled)
   ```
   Isso diz: "comece as previsões a partir deste índice", então só fazemos previsões no período de validação, não no período de treino.

**Alternativa** (que NÃO funciona com `historical_forecasts`):
- Usar `model.predict()` diretamente, mas isso só faz UMA previsão e não permite avaliar o modelo em múltiplos pontos do tempo
- `historical_forecasts` é o método recomendado para backtesting e avaliação robusta

#### Por que `.pd_dataframe()` ao invés de `.values()`?

O Darts `TimeSeries` internamente armazena dados como um DataFrame pandas. O método:
- `.values()` retorna um array numpy 2D que pode ter dimensões inesperadas
- `.pd_dataframe()` retorna o DataFrame pandas subjacente, que é mais previsível e manipulável

#### Por que concatenar séries para `historical_forecasts`?

O método `historical_forecasts` do Darts simula previsões ao longo do tempo. Para cada ponto de previsão em `t`, ele precisa:
1. Acesso aos últimos `input_chunk_length` pontos antes de `t`
2. Acesso às covariáveis para o mesmo período

Quando passamos apenas a série de validação/teste, não há dados históricos suficientes antes do primeiro ponto, causando falha nas previsões.

#### Alternativa: usar `.predict()` ao invés de `.historical_forecasts()`

Uma alternativa seria usar `model.predict(n=OUTPUT_CHUNK)` diretamente, mas:
- `historical_forecasts` é mais robusto para avaliação
- Permite avaliar o modelo em múltiplos horizontes de previsão
- Simula melhor a performance em produção

### Referências

- Documentação Darts TimeSeries: https://unit8co.github.io/darts/generated_api/darts.timeseries.html
- Documentação TCNModel: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html
- TCN_knowledge.md: Base de conhecimento técnica sobre TCN no Darts
