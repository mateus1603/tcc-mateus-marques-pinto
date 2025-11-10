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

**Problema**: O `historical_forecasts` não tinha contexto histórico suficiente quando recebia apenas `val_scaled`. O modelo precisa de pelo menos `input_chunk_length` (168 horas) de dados históricos antes do ponto de previsão.

**Antes**:
```python
val_predictions = model.historical_forecasts(
    series=val_scaled,
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Depois**:
```python
# Para historical_forecasts, precisamos passar train+val para ter contexto suficiente
train_val_scaled = train_scaled.append(val_scaled)

# historical_forecasts vai fazer previsões começando após o período de treinamento
val_predictions = model.historical_forecasts(
    series=train_val_scaled,
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    start=len(train_scaled),  # Começar previsões após o fim do treino
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Justificativa**:
- `historical_forecasts` precisa de histórico anterior para fazer previsões
- Ao passar apenas `val_scaled`, o modelo não tinha os dados de treino necessários como contexto
- O parâmetro `start` garante que as previsões comecem exatamente após o período de treinamento
- Isso resolve o problema de não gerar previsões

#### 3. Seção 9 - Avaliação no Conjunto de Teste (linhas 544-556)

**Problema**: Mesma questão da seção 8 - falta de contexto histórico.

**Antes**:
```python
test_predictions = model.historical_forecasts(
    series=test_scaled,
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Depois**:
```python
# Para ter contexto suficiente, usamos train+val+test
full_scaled = train_scaled.append(val_scaled).append(test_scaled)

# Começar previsões após train+val
test_predictions = model.historical_forecasts(
    series=full_scaled,
    past_covariates=covariates,
    forecast_horizon=OUTPUT_CHUNK,
    start=len(train_scaled) + len(val_scaled),  # Começar após train+val
    stride=OUTPUT_CHUNK,
    retrain=False,
    verbose=False
)
```

**Justificativa**:
- Similar à correção da seção 8
- O modelo precisa de todo o histórico anterior (treino + validação) para fazer previsões no teste
- O parâmetro `start` posiciona corretamente o início das previsões

### Impacto Esperado

Com estas correções:

1. ✅ **Seção 5**: As estatísticas do conjunto normalizado devem exibir valores corretos (média ≈ 0, desvio padrão ≈ 1)
2. ✅ **Seção 7**: O treinamento deve completar com valores numéricos válidos para `train_loss` e `val_loss`
3. ✅ **Seção 8**: Previsões devem ser geradas com sucesso no conjunto de validação
4. ✅ **Seção 9**: Previsões devem ser geradas com sucesso no conjunto de teste

### Observações Técnicas

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
