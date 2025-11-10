# Implementação TCN com Darts - Guia de Migração

## ⚠️ Importante: Nova Implementação Disponível

Foi criada uma nova implementação da TCN utilizando o pacote **Darts**, substituindo a implementação anterior em PyTorch puro. 

## Arquivos Criados

1. **TCN_new.ipynb** - Nova implementação completa usando Darts
2. **utils_TCN_new.py** - Funções utilitárias para a implementação Darts

## Migração

Para utilizar a nova implementação:

### Opção 1: Renomear arquivos (Recomendado)

```bash
# Fazer backup da implementação antiga
mv models/TCN.ipynb models/TCN_pytorch_backup.ipynb
mv models/utils_TCN.py models/utils_TCN_pytorch_backup.py

# Ativar nova implementação
mv models/TCN_new.ipynb models/TCN.ipynb
mv models/utils_TCN_new.py models/utils_TCN.py
```

### Opção 2: Usar diretamente os arquivos novos

Simplesmente abra e execute `models/TCN_new.ipynb`.

## Principais Diferenças

### Implementação Antiga (PyTorch Puro)
- ❌ Implementação manual de toda a arquitetura TCN
- ❌ Código mais complexo e verboso
- ❌ Maior chance de erros de implementação
- ❌ Mais difícil de manter e estender

### Nova Implementação (Darts)
- ✅ Usa a biblioteca Darts (built on PyTorch Lightning)
- ✅ Código mais limpo e profissional
- ✅ Implementação TCN validada e otimizada
- ✅ Melhor documentação e explicações
- ✅ Segue as melhores práticas do TCN paper
- ✅ Suporte nativo para:
  - Covariáveis temporais
  - Early stopping
  - Validação temporal correta
  - Métricas de avaliação
  - Visualizações

## Funcionalidades da Nova Implementação

### 1. Carregamento de Dados
```python
from darts import TimeSeries

# Converter DataFrame para TimeSeries
ts = TimeSeries.from_dataframe(
    df,
    time_col='din_instante',
    value_cols='val_intercambiomwmed',
    freq='H'
)
```

### 2. Covariáveis Temporais
```python
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Criar features de hora, dia da semana, mês
hour_cov = datetime_attribute_timeseries(ts, attribute='hour', one_hot=True)
dow_cov = datetime_attribute_timeseries(ts, attribute='dayofweek', one_hot=True)
month_cov = datetime_attribute_timeseries(ts, attribute='month', one_hot=True)

# Combinar covariáveis
covariates = hour_cov.stack(dow_cov).stack(month_cov)
```

### 3. Modelo TCN
```python
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping

# Configurar modelo
model = TCNModel(
    input_chunk_length=168,   # 1 semana de dados
    output_chunk_length=24,   # Previsão de 1 dia
    kernel_size=3,
    num_filters=64,
    num_layers=4,
    dilation_base=2,
    weight_norm=True,
    dropout=0.2,
    n_epochs=100,
    batch_size=32,
    pl_trainer_kwargs={
        'callbacks': [EarlyStopping(monitor="val_loss", patience=10)]
    }
)

# Treinar
model.fit(
    series=train_scaled,
    past_covariates=covariates,
    val_series=val_scaled,
    val_past_covariates=covariates
)
```

### 4. Previsão
```python
# Fazer previsão
forecast = model.predict(
    n=24,  # 24 horas
    series=train_scaled,
    past_covariates=covariates
)

# Inverter normalização
forecast_original = scaler.inverse_transform(forecast)
```

### 5. Avaliação
```python
from darts.metrics import mae, rmse, mape

# Calcular métricas
mae_value = mae(actual, predicted)
rmse_value = rmse(actual, predicted)
mape_value = mape(actual, predicted)
```

## Cálculo do Campo Receptivo (Receptive Field)

A nova implementação inclui cálculo automático do RF:

```python
from utils_TCN import calculate_receptive_field

rf = calculate_receptive_field(kernel_size=3, num_layers=4, dilation_base=2)
# RF = 31 horas
```

**Importante**: O `input_chunk_length` deve ser >= RF para o modelo utilizar sua memória completa.

## Estrutura do Notebook

1. **Imports e Configuração**
2. **Funções Auxiliares** (cálculo de RF, visualizações)
3. **Carregamento e Preparação de Dados**
4. **Divisão Temporal** (Train/Val/Test)
5. **Criação de Covariáveis Temporais**
6. **Normalização** (sem vazamento de dados)
7. **Configuração do Modelo TCN**
8. **Treinamento** (com early stopping)
9. **Avaliação** (validação e teste)
10. **Previsão Futura**
11. **Análise de Resultados** (métricas, resíduos)
12. **Conclusões**

## Parâmetros Recomendados

Para dados horários de intercâmbio de energia:

- **input_chunk_length**: 168 (1 semana)
- **output_chunk_length**: 24 (1 dia) ou 168 (1 semana)
- **kernel_size**: 3
- **num_layers**: 4-6 (RF de 31-127 horas)
- **num_filters**: 64-128
- **dilation_base**: 2 (padrão)
- **dropout**: 0.1-0.3

## Vantagens da Abordagem Darts

1. **Qualidade**: Implementação revisada e testada pela comunidade
2. **Manutenibilidade**: Código mais simples e legível
3. **Extensibilidade**: Fácil testar outros modelos (RNN, Transformer, N-BEATS)
4. **Performance**: Otimizações integradas do PyTorch Lightning
5. **Debugging**: Melhor logging e monitoramento
6. **Documentação**: Extensa documentação oficial do Darts

## Limitações Conhecidas

- **TCNModel** do Darts suporta apenas `past_covariates` (não `future_covariates`)
- Não suporta `static_covariates` nativamente
- Para esses casos, considere usar `TFTModel` (Temporal Fusion Transformer)

## Referências

- [Darts Documentation](https://unit8co.github.io/darts/)
- [TCN Paper](https://arxiv.org/abs/1803.01271) - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- [TCN_knowledge.md](../data/TCN_knowledge.md) - Guia detalhado da implementação

## Próximos Passos

1. Executar `TCN_new.ipynb` para verificar funcionamento
2. Ajustar hiperparâmetros conforme necessário
3. Experimentar com diferentes horizontes de previsão
4. Comparar com outros modelos Darts (RNN, Transformer, etc.)
5. Implementar otimização de hiperparâmetros

## Suporte

Em caso de dúvidas sobre a implementação, consulte:
- `data/TCN_knowledge.md` - Guia técnico completo
- `data/dataset_description.md` - Descrição dos dados
- [Darts Examples](https://github.com/unit8co/darts/tree/master/examples) - Exemplos oficiais
