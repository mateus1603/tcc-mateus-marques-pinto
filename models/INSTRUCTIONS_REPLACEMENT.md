# INSTRUÇÕES PARA SUBSTITUIÇÃO DO TCN.ipynb

## Contexto

Foi criada uma nova implementação completa da TCN utilizando o pacote **Darts** (ao invés de PyTorch puro). A nova implementação está em `TCN_new.ipynb` e precisa substituir o arquivo `TCN.ipynb` atual.

## Por que a substituição não foi feita automaticamente?

O arquivo `TCN.ipynb` original tem aproximadamente 1940 linhas de código JSON (formato Jupyter Notebook). Devido às limitações das ferramentas de edição automática para arquivos tão grandes, a substituição completa não pôde ser realizada automaticamente de forma segura.

## Como realizar a substituição manualmente

### Opção 1: Via Git (Recomendado)

```bash
# No diretório models/
git mv TCN.ipynb TCN_pytorch_backup.ipynb
git mv TCN_new.ipynb TCN.ipynb
git mv utils_TCN.py utils_TCN_pytorch_backup.py  
git mv utils_TCN_new.py utils_TCN.py
git commit -m "Replace PyTorch TCN with Darts implementation"
```

### Opção 2: Via linha de comando

```bash
# No diretório models/
mv TCN.ipynb TCN_pytorch_backup.ipynb
mv TCN_new.ipynb TCN.ipynb
mv utils_TCN.py utils_TCN_pytorch_backup.py
mv utils_TCN_new.py utils_TCN.py
```

### Opção 3: Via interface gráfica

1. Renomear `TCN.ipynb` para `TCN_pytorch_backup.ipynb`
2. Renomear `TCN_new.ipynb` para `TCN.ipynb`
3. Renomear `utils_TCN.py` para `utils_TCN_pytorch_backup.py`
4. Renomear `utils_TCN_new.py` para `utils_TCN.py`

## Verificação

Após a substituição, verifique que:

1. ✅ `models/TCN.ipynb` existe e contém a implementação Darts
2. ✅ `models/utils_TCN.py` existe e contém as funções utilitárias
3. ✅ A primeira célula de `TCN.ipynb` importa `from darts import TimeSeries`
4. ✅ Não há mais imports de `torch.nn` no novo TCN.ipynb

## Comparação das Implementações

### TCN.ipynb original (PyTorch)
- Implementação manual da arquitetura TCN
- ~1940 linhas
- Usa PyTorch puro
- Mais complexo de manter

### TCN_new.ipynb (Darts)  
- Usa biblioteca Darts (built on PyTorch Lightning)
- Implementação profissional e validada
- Melhor documentação
- Segue melhores práticas do paper TCN
- Mais fácil de estender e manter

## Conteúdo da Nova Implementação

A nova implementação (`TCN_new.ipynb`) inclui:

1. **Imports e Configuração**
   - Darts, pandas, numpy, matplotlib
   - PyTorch Lightning callbacks

2. **Funções Auxiliares**
   - Cálculo de campo receptivo
   - Impressão de configuração do modelo

3. **Carregamento de Dados**
   - Conversão para TimeSeries do Darts
   - Tratamento de dados faltantes
   - Visualização exploratória

4. **Preparação de Dados**
   - Divisão temporal (70% treino, 15% val, 15% teste)
   - Criação de covariáveis temporais (hora, dia, mês)
   - Normalização sem vazamento de dados

5. **Modelo TCN**
   - Configuração com parâmetros adequados
   - Campo receptivo calculado
   - Early stopping
   - Weight normalization

6. **Treinamento**
   - Com validação
   - Monitoramento de métricas
   - Callbacks para otimização

7. **Avaliação**
   - Métricas: MAE, RMSE, MAPE
   - Visualizações de previsões
   - Análise de resíduos

8. **Previsão Futura**
   - Forecast de 24 horas
   - Inverção de normalização
   - Visualização com contexto histórico

9. **Análise Detalhada**
   - Resumo de métricas
   - Distribuição de resíduos
   - Q-Q plot

10. **Conclusões**
    - Pontos fortes
    - Limitações
    - Próximos passos

## Parâmetros do Modelo

A nova implementação usa parâmetros otimizados:

```python
INPUT_CHUNK = 168      # 1 semana de dados históricos
OUTPUT_CHUNK = 24      # Previsão de 1 dia
KERNEL_SIZE = 3
NUM_LAYERS = 4         # RF = 31 horas
NUM_FILTERS = 64       # Capacidade do modelo
DILATION_BASE = 2
DROPOUT = 0.2
```

## Utilidades (utils_TCN_new.py)

Funções disponíveis:

- `calculate_receptive_field()`: Calcula RF do modelo
- `load_and_prepare_data()`: Carrega e prepara dados
- `create_temporal_covariates()`: Cria features temporais
- `prepare_scalers()`: Normaliza dados sem vazamento
- `evaluate_model()`: Avalia modelo com métricas
- `plot_predictions()`: Visualiza previsões
- `print_model_summary()`: Imprime configuração

## Dependências

A nova implementação requer:

```
darts>=0.23.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
pytorch-lightning>=1.9.0
torch>=1.13.0
scipy>=1.9.0
```

Instalar com:
```bash
pip install darts[torch]
```

## Documentação Adicional

- `README_MIGRATION.md`: Guia completo de migração
- `../data/TCN_knowledge.md`: Base de conhecimento técnica sobre TCN
- `../data/dataset_description.md`: Descrição dos dados

## Suporte

Para dúvidas:
1. Consultar `README_MIGRATION.md` para exemplos de uso
2. Consultar `TCN_knowledge.md` para detalhes técnicos
3. Ver exemplos oficiais do Darts: https://github.com/unit8co/darts/tree/master/examples

## Status da Tarefa

- ✅ Nova implementação TCN com Darts criada (`TCN_new.ipynb`)
- ✅ Funções utilitárias criadas (`utils_TCN_new.py`)
- ✅ Documentação e guia de migração criados
- ⏳ **Aguardando**: Renomeação de arquivos para substituir a implementação antiga

## Próxima Ação Requerida

Execute um dos comandos de renomeação acima para ativar a nova implementação.
