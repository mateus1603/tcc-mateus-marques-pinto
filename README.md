# TCC - Mateus Marques Pinto

## Repositório do Trabalho de Conclusão de Curso

Este repositório contém todos os arquivos e códigos relacionados ao desenvolvimento do TCC sobre análise de dados de intercâmbio nacional de energia elétrica.

## 📁 Estrutura do Projeto

```text
📦 tcc-mateus-marques-pinto
 ┣ 📂 data/
 ┃ ┗ 📄 ONS-data.ipynb          # Download e preparação dos dados do ONS
 ┣ 📂 eda/
 ┃ ┗ 📄 EDA.ipynb               # Análise Exploratória de Dados
 ┣ 📂 models/
 ┃ ┗ (em desenvolvimento)       # Modelos de machine learning
 ┣ 📄 .gitignore                # Configurações de exclusão do Git
 ┗ 📄 README.md                 # Este arquivo
```

## 🔗 Fonte dos Dados

Os dados utilizados são provenientes do **ONS (Operador Nacional do Sistema Elétrico)**, especificamente do dataset de intercâmbio nacional disponível em:

- <https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/intercambio_nacional_ho/>

## 📊 Notebooks

### 1. ONS-data.ipynb

- **Localização**: `data/ONS-data.ipynb`
- **Objetivo**: Download automático e preparação inicial dos dados
- **Funcionalidades**:
  - Download de dados dos anos 2023, 2024 e 2025
  - Consolidação em arquivo único
  - Limpeza e padronização inicial
  - Criação de features de calendário

### 2. EDA.ipynb

- **Localização**: `eda/EDA.ipynb`
- **Objetivo**: Análise Exploratória de Dados
- **Funcionalidades**:
  - Análise de padrões temporais
  - Visualizações interativas
  - Estatísticas descritivas
  - Identificação de tendências

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Plotly** - Visualizações interativas
- **Jupyter Notebook** - Ambiente de desenvolvimento

## 🚀 Como Usar

1. Clone o repositório:

   ```bash
   git clone https://github.com/mateus1603/tcc-mateus-marques-pinto.git
   ```

2. Navegue até o diretório:

   ```bash
   cd tcc-mateus-marques-pinto
   ```

3. Execute os notebooks na seguinte ordem:
   - Primeiro: `data/ONS-data.ipynb` (para baixar e preparar os dados)
   - Segundo: `eda/EDA.ipynb` (para análise exploratória)

## 📝 Status do Projeto

- ✅ Coleta e preparação de dados
- ✅ Análise exploratória de dados
- 🔄 Desenvolvimento de modelos (em andamento)
- ⏳ Análise de resultados (pendente)
- ⏳ Documentação final (pendente)

## 👨‍🎓 Autor

### Mateus Marques Pinto

- GitHub: [@mateus1603](https://github.com/mateus1603)

---

*Este projeto faz parte do Trabalho de Conclusão de Curso (TCC) e tem fins acadêmicos.*
