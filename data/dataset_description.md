# Dicionário de Dados - Intercâmbio Entre Subsistemas

**Versão:** 1.1  
**Data:** 13/09/2022  
**Fonte:** ONS (Operador Nacional do Sistema Elétrico)

---

## 📋 Descrição do Dataset

Este dataset contém **dados de intercâmbio entre subsistemas** do Sistema Interligado Nacional (SIN) em base horária, medidos em **MWmed** (Megawatts médios).

### O que representa
As grandezas representam a **soma das medidas de fluxo de potência ativa** nas linhas de transmissão de fronteira entre os subsistemas elétricos brasileiros.

---

## ⚠️ Observações Importantes

1. **Linhas de Fronteira:** A relação completa de linhas de transmissão de fronteira pode ser encontrada no produto "Relatório Quadrimestral de Limites de Intercâmbio para o Modelo Newave", disponível no Portal SINtegre - ONS.

2. **Intercâmbio Internacional:** O intercâmbio do subsistema Sul com os países vizinhos **NÃO** consta nesta consulta. Esses dados podem ser obtidos separadamente nos dados de intercâmbio do SIN.

---

## 📊 Estrutura dos Dados

| Campo | Código | Tipo | Formato | Nulo | Zero | Negativo |
|-------|--------|------|---------|------|------|----------|
| **Data/hora** (início do período de agregação) | `din_instante` | DATETIME | `YYYY-MM-DD HH:MM:SS` | ❌ Não | - | - |
| **Código do Subsistema de Origem** | `id_subsistema_origem` | TEXTO | 3 posições | ❌ Não | - | - |
| **Nome do Subsistema de Origem** | `nom_subsistema_origem` | TEXTO | 20 posições | ❌ Não | - | - |
| **Código do Subsistema de Destino** | `id_subsistema_destino` | TEXTO | 3 posições | ❌ Não | - | - |
| **Nome do Subsistema de Destino** | `nom_subsistema_destino` | TEXTO | 20 posições | ❌ Não | - | - |
| **Intercâmbio Verificado** | `val_intercambiomwmed` | FLOAT | Numérico | ❌ Não | ✅ Sim | ✅ Sim |

---

## 🔍 Detalhamento dos Campos

### 1. `din_instante` - Data/Hora
- **Propósito:** Marca temporal indicando o início do período de agregação horária
- **Formato:** Ano-Mês-Dia Hora:Minuto:Segundo
- **Exemplo:** `2022-09-13 14:00:00`
- **Obrigatório:** Sim (não permite valores nulos)

### 2. `id_subsistema_origem` - Código de Origem
- **Propósito:** Identificador do subsistema que origina o fluxo de energia
- **Formato:** Código alfanumérico de 3 caracteres
- **Exemplos:** `SE` (Sudeste), `S` (Sul), `NE` (Nordeste), `N` (Norte)
- **Obrigatório:** Sim

### 3. `nom_subsistema_origem` - Nome de Origem
- **Propósito:** Nome completo do subsistema de origem
- **Formato:** Texto com até 20 caracteres
- **Exemplo:** `SUDESTE`, `SUL`, `NORDESTE`, `NORTE`
- **Obrigatório:** Sim

### 4. `id_subsistema_destino` - Código de Destino
- **Propósito:** Identificador do subsistema que recebe o fluxo de energia
- **Formato:** Código alfanumérico de 3 caracteres
- **Obrigatório:** Sim

### 5. `nom_subsistema_destino` - Nome de Destino
- **Propósito:** Nome completo do subsistema de destino
- **Formato:** Texto com até 20 caracteres
- **Obrigatório:** Sim

### 6. `val_intercambiomwmed` - Valor do Intercâmbio
- **Propósito:** Medida do fluxo de potência ativa
- **Unidade:** MWmed (Megawatts médios)
- **Tipo:** Número decimal (float)
- **Valores Permitidos:**
  - ✅ Valores positivos: Indica fluxo no sentido origem → destino
  - ✅ Valores negativos: Indica fluxo no sentido destino → origem
  - ✅ Zero: Indica ausência de intercâmbio no período
  - ❌ Nulo: Não permitido
- **Interpretação:**
  - **Positivo:** Energia fluindo do subsistema de origem para o de destino
  - **Negativo:** Energia fluindo no sentido contrário (de destino para origem)

---

## 🔄 Interpretação dos Valores

### Valores Positivos vs Negativos
O campo `val_intercambiomwmed` pode assumir valores negativos, o que é **esperado e válido**:

- **Valor > 0:** Fluxo no sentido declarado (origem → destino)
- **Valor < 0:** Fluxo no sentido inverso (destino → origem)
- **Valor = 0:** Sem intercâmbio no período

### Exemplo Prático
```
Origem: SUDESTE → Destino: SUL
val_intercambiomwmed = 500 MW  → 500 MW fluindo de SE para S
val_intercambiomwmed = -300 MW → 300 MW fluindo de S para SE
val_intercambiomwmed = 0 MW    → Sem fluxo entre os subsistemas
```

---

## 📈 Aplicações dos Dados

- Análise de fluxo de energia entre regiões do Brasil
- Monitoramento da capacidade de transmissão
- Planejamento operacional do sistema elétrico
- Estudos de segurança energética
- Identificação de padrões sazonais de consumo
- Otimização do despacho de energia

---

## 📝 Histórico de Versões

### Versão 1.0
- Criação inicial do dicionário de dados

### Versão 1.1 (Atual)
- Incorporação de informações para validação e checagem de qualidade
- Documentação sobre valores nulos ou vazios
- Documentação sobre valores numéricos negativos e zero

---

## 🔗 Referências

- **Portal SINtegre - ONS:** Plataforma oficial para consulta de dados do sistema elétrico brasileiro
- **Relatório Quadrimestral de Limites de Intercâmbio:** Documento complementar com detalhes das linhas de transmissão