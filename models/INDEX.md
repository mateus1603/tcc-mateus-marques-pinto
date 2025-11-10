# 📚 Índice da Documentação TCN - Implementação Darts

## 🎯 Início Rápido

**Para usuários que querem começar imediatamente:**

1. Leia: [`INSTRUCTIONS_REPLACEMENT.md`](./INSTRUCTIONS_REPLACEMENT.md)
2. Execute os comandos de renomeação
3. Abra: `TCN.ipynb` (renomeado de `TCN_new.ipynb`)
4. Execute célula por célula

## 📖 Documentos Disponíveis

### 1. Implementação Principal
- **`TCN_new.ipynb`** - Notebook Jupyter com implementação completa
  - 729 linhas de código e markdown
  - 16 seções organizadas
  - Executável célula por célula
  - Todos os comentários em português

### 2. Código Utilitário
- **`utils_TCN_new.py`** - Funções auxiliares Python
  - 323 linhas
  - 7 funções principais
  - Type hints completos
  - Docstrings detalhadas

### 3. Documentação Essencial

#### Para Migração
- **`INSTRUCTIONS_REPLACEMENT.md`** ⭐ **COMECE AQUI**
  - Como substituir a implementação antiga
  - 3 métodos diferentes (Git, CLI, GUI)
  - Checklist de verificação
  - ~250 linhas

#### Para Aprendizado
- **`README_MIGRATION.md`**
  - Guia completo de migração
  - Comparação PyTorch vs Darts
  - Exemplos de código
  - Parâmetros recomendados
  - ~300 linhas

#### Para Referência Técnica
- **`COMPLETE_SUMMARY.md`** ⭐ **REFERÊNCIA COMPLETA**
  - Resumo técnico abrangente
  - Fundamentos da arquitetura TCN
  - Guia de ajuste de hiperparâmetros
  - Próximos passos sugeridos
  - ~500 linhas

### 4. Documentação Original do Projeto
- **`../data/TCN_knowledge.md`** - Base de conhecimento técnica sobre TCN
  - Teoria detalhada
  - Paper original
  - Implementação no Darts
  
- **`../data/dataset_description.md`** - Descrição dos dados ONS
  - Formato dos dados
  - Significado das colunas
  - Subsistemas elétricos

## 🗺️ Fluxo de Uso Recomendado

### Primeira Vez (Setup)

```
1. INSTRUCTIONS_REPLACEMENT.md
   ↓
2. Executar comandos de renomeação
   ↓
3. Instalar dependências (pip install darts[torch])
   ↓
4. Executar ../data/ONS-data.ipynb (se necessário)
   ↓
5. Abrir TCN.ipynb
```

### Entendimento da Implementação

```
1. COMPLETE_SUMMARY.md (ler seções: Visão Geral, Arquitetura TCN)
   ↓
2. TCN.ipynb (executar e ler comentários)
   ↓
3. ../data/TCN_knowledge.md (teoria detalhada)
```

### Uso Prático

```
1. TCN.ipynb (executar)
   ↓
2. Ajustar parâmetros
   ↓
3. COMPLETE_SUMMARY.md (seção: Parâmetros Recomendados)
   ↓
4. Experimentar
```

### Troubleshooting

```
1. README_MIGRATION.md (seção: Funcionalidades)
   ↓
2. COMPLETE_SUMMARY.md (seção: Ajuste Fino)
   ↓
3. ../data/TCN_knowledge.md (seção: Limitações)
```

## 📊 Estrutura dos Documentos

### INSTRUCTIONS_REPLACEMENT.md
```
├── Contexto
├── Por que substituição não foi automática
├── Como realizar substituição (3 métodos)
├── Verificação
├── Comparação das implementações
├── Conteúdo da nova implementação
├── Parâmetros do modelo
├── Utilidades (utils_TCN_new.py)
├── Dependências
├── Documentação adicional
└── Próxima ação requerida
```

### README_MIGRATION.md
```
├── Importante: Nova implementação disponível
├── Arquivos criados
├── Migração (2 opções)
├── Principais diferenças
├── Funcionalidades (6 seções com código)
├── Cálculo do campo receptivo
├── Estrutura do notebook
├── Parâmetros recomendados
├── Vantagens da abordagem Darts
├── Limitações conhecidas
├── Referências
└── Próximos passos
```

### COMPLETE_SUMMARY.md
```
├── Visão geral
├── Objetivos alcançados
├── Arquivos criados (descrição detalhada)
├── Fundamentos técnicos
│   ├── Arquitetura TCN
│   ├── Cálculo do campo receptivo
│   ├── Estratégia de previsão
│   ├── Prevenção de vazamento
│   └── Covariáveis temporais
├── Avaliação e métricas
├── Como usar (5 passos)
├── Parâmetros recomendados
├── Conceitos importantes
├── Referências e recursos
├── Checklist de verificação
└── Próximos passos sugeridos
```

## 🎓 Por Nível de Experiência

### Iniciante
1. Leia `INSTRUCTIONS_REPLACEMENT.md` (seções 1-4)
2. Execute os comandos
3. Abra `TCN.ipynb` e execute célula por célula
4. Leia os comentários no notebook
5. Consulte `README_MIGRATION.md` para entender funcionalidades

### Intermediário
1. Leia `COMPLETE_SUMMARY.md` (seções: Visão Geral, Arquitetura)
2. Execute `TCN.ipynb` completo
3. Experimente ajustar parâmetros do modelo
4. Leia `../data/TCN_knowledge.md` para entender teoria
5. Use funções de `utils_TCN_new.py` nos seus experimentos

### Avançado
1. Leia `COMPLETE_SUMMARY.md` completo
2. Estude `utils_TCN_new.py` (código fonte)
3. Leia `../data/TCN_knowledge.md` (seções técnicas)
4. Implemente otimização de hiperparâmetros
5. Compare com outros modelos Darts (RNN, Transformer, N-BEATS)

## 🔍 Busca Rápida por Tópico

### Instalação e Setup
- `INSTRUCTIONS_REPLACEMENT.md` → seção "Dependências"
- `README_MIGRATION.md` → seção "Migração"

### Teoria da TCN
- `COMPLETE_SUMMARY.md` → seção "Fundamentos Técnicos"
- `../data/TCN_knowledge.md` → completo

### Parâmetros do Modelo
- `COMPLETE_SUMMARY.md` → seção "Parâmetros Recomendados"
- `TCN.ipynb` → célula "Configuração do Modelo TCN"

### Campo Receptivo
- `COMPLETE_SUMMARY.md` → seção "Cálculo do Campo Receptivo"
- `utils_TCN_new.py` → função `calculate_receptive_field()`

### Covariáveis Temporais
- `COMPLETE_SUMMARY.md` → seção "Covariáveis Temporais"
- `TCN.ipynb` → seção 4
- `utils_TCN_new.py` → função `create_temporal_covariates()`

### Avaliação e Métricas
- `COMPLETE_SUMMARY.md` → seção "Avaliação e Métricas"
- `TCN.ipynb` → seções 8-11
- `utils_TCN_new.py` → função `evaluate_model()`

### Troubleshooting
- `README_MIGRATION.md` → seção "Limitações conhecidas"
- `COMPLETE_SUMMARY.md` → seção "Ajuste Fino"

### Próximos Passos
- `COMPLETE_SUMMARY.md` → seção "Próximos Passos Sugeridos"
- `TCN.ipynb` → seção 12 (Conclusões)

## 📞 FAQ - Perguntas Frequentes

**P: Por que TCN_new.ipynb e não TCN.ipynb diretamente?**
R: O arquivo TCN.ipynb original tem 1940 linhas e não pôde ser substituído automaticamente. Veja `INSTRUCTIONS_REPLACEMENT.md`.

**P: Preciso da implementação PyTorch antiga?**
R: Não, mas ela será mantida como backup (TCN_pytorch_backup.ipynb).

**P: Quais dados usar?**
R: Execute `../data/ONS-data.ipynb` primeiro para baixar dados de 2022-2025.

**P: Posso usar outros modelos Darts?**
R: Sim! O código é facilmente adaptável para RNNModel, TransformerModel, TFTModel, etc.

**P: Como ajusto hiperparâmetros?**
R: Veja seção "Parâmetros Recomendados" em `COMPLETE_SUMMARY.md`.

**P: O que é campo receptivo?**
R: Quantos passos de tempo no passado o modelo pode "ver". Veja `COMPLETE_SUMMARY.md` → "Cálculo do Campo Receptivo".

**P: Por que one-shot forecasting?**
R: É mais rápido e não propaga erros. Veja `COMPLETE_SUMMARY.md` → "Estratégia de Previsão".

**P: Posso usar future_covariates?**
R: TCNModel não suporta. Use TFTModel para isso. Veja `README_MIGRATION.md` → "Limitações".

## 🛠️ Arquivos de Código

### Para Importar
```python
# Funções utilitárias
from utils_TCN import (
    calculate_receptive_field,
    load_and_prepare_data,
    create_temporal_covariates,
    prepare_scalers,
    evaluate_model,
    plot_predictions,
    print_model_summary
)
```

### Para Executar
- `TCN.ipynb` (após renomeação)
- `../data/ONS-data.ipynb` (para baixar dados)

## 📈 Status do Projeto

- ✅ Implementação completa
- ✅ Documentação abrangente
- ✅ Código testado (estruturalmente)
- ⏳ Execução com dados reais (depende do usuário)
- ⏳ Otimização de hiperparâmetros (próximos passos)

## 🔗 Links Úteis

### Darts
- [Documentação oficial](https://unit8co.github.io/darts/)
- [GitHub](https://github.com/unit8co/darts)
- [Exemplos](https://github.com/unit8co/darts/tree/master/examples)
- [Paper TCN](https://arxiv.org/abs/1803.01271)

### PyTorch Lightning
- [Documentação](https://pytorch-lightning.readthedocs.io/)
- [Callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)

## 📝 Histórico de Versões

### v1.0 (2025-11-10)
- ✅ Implementação inicial completa
- ✅ 5 documentos criados
- ✅ 729 linhas de código no notebook
- ✅ 323 linhas de utilitários
- ✅ ~1300 linhas de documentação

---

**Última atualização:** 2025-11-10  
**Mantenedor:** Copilot Coding Agent  
**Licença:** Conforme repositório principal
