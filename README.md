# Deteção de Autoria em Textos Científicos (IA vs Humano)

### **Universidade Katyavala Bwila — Instituto Politécnico**

#### **Curso de Mestrado em Engenharia Informática**

**Unidade Curricular:** Aprendizagem Profunda — Ano Lectivo 2025/2026  
**Docentes:** Prof. Doutor Miguel Rocha & Prof. Doutor Lázaro Makili  

### **Grupo de Trabalho nº 6**

**Integrantes do Grupo:**

- **Estanislau da Fonseca**
- **Maria da Conceição**
- **Maurício Piquito**
- **Ulundo Aurélio Moisés**

---

## Visão Geral do Projeto

Este repositório contém a solução integral desenvolvida no âmbito do trabalho prático da unidade curricular de **Aprendizagem Profunda**. O objetivo central é o desenvolvimento e avaliação de modelos de classificação binária capazes de discriminar textos em inglês escritos por humanos de textos gerados por Modelos de Linguagem de Grande Escala (LLMs), circunscritos ao domínio das **Ciências Naturais e Médicas** (Biologia, Saúde, Química, Física, Matemática e Engenharia) com restrição estrita de comprimento (**100 a 120 palavras**).

Todos os artefactos, modelos pré-treinados, relatórios e pipelines de inferência foram estruturados de forma modular e reprodutível para facilitar a avaliação direta por parte do corpo docente.

---

## Guia Rápido para o Docente (Avaliação e Submissões)

Para avaliar novos dados ou inspecionar as submissões sem necessidade de novo treino:

### Opção 1: Notebook Interativo de Submissão (Recomendado)

1. Abrir o notebook [`notebooks/tarefa4_submissao_docente.ipynb`](notebooks/tarefa4_submissao_docente.ipynb).
2. Definir o caminho do ficheiro de teste do docente (`.csv` ou `.tsv` com colunas `ID` e `text`).
3. Executar as células sequencialmente. O pipeline carrega os pesos em `models/` e gera automaticamente as predições.

### Opção 2: Ficheiros de Submissão Pré-Gerados

Os ficheiros prontos para submissão (gerados sobre o conjunto de validação simulado) encontram-se na pasta `notebooks/`:

- **1ª Submissão (Modelo NumPy de Raiz + Modelo PyTorch):**
  - [`notebooks/submissao_fase1_numpy.csv`](notebooks/submissao_fase1_numpy.csv) — *NumPy Logistic Regression (Baseline)*
  - [`notebooks/submissao_fase1_pytorch.csv`](notebooks/submissao_fase1_pytorch.csv) — *PyTorch BiLSTM (Embeddings)*
- **2ª Submissão (Dois Melhores Modelos Globais):**
  - [`notebooks/submissao_fase2_top1.csv`](notebooks/submissao_fase2_top1.csv) — *TOP 1: PyTorch BiLSTM (F1: 66.67%, ROC-AUC: 0.8333)*
  - [`notebooks/submissao_fase2_top2.csv`](notebooks/submissao_fase2_top2.csv) — *TOP 2: PyTorch Tabular DNN TF-IDF (F1: 62.50%, ROC-AUC: 0.8095)*

---

## Estrutura do Repositório

```text
nosso_trabalho_ap_v_final_v2/
├── README.md                          # Documentação principal e guia de navegação
├── relatorio_final_artigo.pdf         # Relatório científico final formatado (Artigo completo)
├── relatorio_final_artigo.md          # Versão integral em Markdown do artigo científico
│
├── data/                              # Datasets brutos, conjuntos limpos e dados de validação
│   ├── dataset1_inputs.csv            # Textos brutos do primeiro lote de dados
│   ├── dataset1_outputs.csv           # Rótulos (Human vs AI) do primeiro lote
│   ├── dataset2_complete.csv          # Segundo lote de dados consolidado
│   ├── train_clean.csv                # Conjunto de treino limpo e estratificado (80%)
│   ├── val_clean.csv                  # Conjunto de validação balanceado (10%)
│   ├── test_clean.csv                 # Conjunto de teste independente não-visto (10%)
│   └── simulated_teacher_validation.csv # Amostra para validação do pipeline do docente
│
├── models/                            # Pesos e pré-processadores serializados
│   ├── best_numpy_lr.pkl              # Pesos da Regressão Logística (NumPy de raiz)
│   ├── best_numpy_dnn.pkl             # Pesos da Rede Neuronal Profunda / MLP (NumPy de raiz)
│   ├── best_pytorch_bilstm.pt         # Pesos da Bidirectional LSTM (PyTorch) — TOP #1 Global
│   ├── best_pytorch_dnn.pt            # Pesos da Tabular DNN TF-IDF (PyTorch) — TOP #2 Global
│   ├── best_pytorch_transformer.pt    # Pesos do Transformer Encoder (PyTorch)
│   ├── tfidf_vectorizer.pkl           # Vetorizador TF-IDF ajustado
│   └── simple_tokenizer.pkl           # Tokenizador e vocabulário sequencial
│
├── notebooks/                         # Notebooks de experimentação, inspeção e submissão
│   ├── tarefa4_submissao_docente.ipynb# Notebook principal de submissão e inferência do docente
│   ├── tarefa3_pytorch.ipynb          # Treino, ajuste de hiperparâmetros e gráficos em PyTorch
│   ├── inspecionar_modelos.ipynb      # Inspeção interativa dos pesos, vocabulários e matrizes
│   ├── submissao_fase1_numpy.csv      # Ficheiro de predição Fase 1 (NumPy)
│   ├── submissao_fase1_pytorch.csv    # Ficheiro de predição Fase 1 (PyTorch)
│   ├── submissao_fase2_top1.csv       # Ficheiro de predição Fase 2 (TOP 1 Global)
│   └── submissao_fase2_top2.csv       # Ficheiro de predição Fase 2 (TOP 2 Global)
│
├── results/                           # Gráficos de benchmark, curvas ROC e matrizes de confusão
│   ├── benchmark_ranking_table.csv    # Tabela consolidada com todas as métricas de avaliação
│   ├── benchmark_metrics_comparison.png # Gráfico de barras comparativo de desempenho
│   ├── roc_curves_all_models.png      # Curvas ROC e áreas sob a curva (ROC-AUC)
│   ├── confusion_matrices_all_models.png # Matrizes de confusão dos 5 modelos
│   └── latency_vs_accuracy.png        # Análise de trade-off entre Latência (ms) e F1-Score
│
├── src/                               # Módulos em Python (Framework de raiz e Inferência)
│   ├── activation.py                  # Sigmoid, ReLU, Tanh, Softmax e respetivas derivadas
│   ├── data.py                        # Carregamento, batching e pré-processamento de datasets
│   ├── layers.py                      # Camada DenseLayer com forward e backward pass manuais
│   ├── losses.py                      # Funções de perda (Binary Cross-Entropy e MSE) e gradientes
│   ├── metrics.py                     # Cálculo manual de Accuracy, Precision, Recall e F1
│   ├── neuralnetwork.py               # Classe NeuralNetwork em NumPy puro com fit(), predict(), evaluate()
│   ├── optimizer.py                   # Otimizadores manuais (SGD com Momentum e Adam)
│   └── inference.py                   # Motor unificado ModelInferenceEngine para execução rápida
│
├── tarefa1_construcao_data.py         # Tarefa 1: Script de curadoria, limpeza e divisão do dataset
├── tarefa2_numpay.py                  # Tarefa 2: Treino e avaliação dos modelos de raiz em NumPy
├── tarefa3_pytorch.py                 # Tarefa 3: Treino e avaliação dos modelos de Deep Learning em PyTorch
├── tarefa4_avaliacao_modelos.py       # Tarefa 4: Benchmark independente, métricas e gráficos comparativos
└── inspecionar_modelos.py             # Script utilitário de diagnóstico e inspeção de artefactos
```

---

## Tabela de Benchmark Comparativo (Resultados Obtidos)

Avaliação realizada no conjunto de teste independente (`data/test_clean.csv`):

| Rank | Modelo | Framework / Abordagem | Exatidão | F1-Score | Precisão | Recall | ROC-AUC | Latência / Amostra |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **PyTorch BiLSTM (Embeddings)** | PyTorch | **69.23%** | **66.67%** | **66.67%** | **66.67%** | **0.8333** | 8.96 ms |
| **#2** | **PyTorch Tabular DNN (TF-IDF)** | PyTorch | 53.85% | 62.50% | 50.00% | **83.33%** | 0.8095 | 12.82 ms |
| **#3** | **PyTorch Transformer Encoder** | PyTorch | 38.46% | 50.00% | 40.00% | 66.67% | 0.6190 | 8.97 ms |
| **#4** | **NumPy Logistic Reg (Baseline)** | NumPy Puro (de raiz) | 46.15% | 36.36% | 40.00% | 33.33% | 0.4286 | 2.43 ms |
| **#5** | **NumPy DNN / MLP** | NumPy Puro (de raiz) | 38.46% | 33.33% | 33.33% | 33.33% | 0.4286 | **1.57 ms** |

---

## Como Executar as Tarefas Localmente

### 1. Pré-requisitos e Dependências

Recomenda-se a utilização de um ambiente virtual Python (3.9+):

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### 2. Execução dos Scripts das Tarefas

- **Tarefa 1 — Preparação do Dataset:**

  ```bash
  python tarefa1_construcao_data.py
  ```

- **Tarefa 2 — Treino dos Modelos de Raiz em NumPy Puro:**

  ```bash
  python tarefa2_numpay.py
  ```

- **Tarefa 3 — Treino dos Modelos Deep Learning em PyTorch:**

  ```bash
  python tarefa3_pytorch.py
  ```

- **Tarefa 4 — Benchmark Completo, Curvas ROC e Matrizes de Confusão:**

  ```bash
  python tarefa4_avaliacao_modelos.py
  ```

- **Inspeção de Modelos e Diagnóstico:**

  ```bash
  python inspecionar_modelos.py
  ```

---

## Relatório Final

O artigo científico completo que sintetiza todo o trabalho desenvolvido, fundamentação teórica, análise de erros e conclusões está disponível em:

- Documento formatado pronto para leitura: [`relatorio_final_artigo.pdf`](relatorio_final_artigo.pdf)
- Código-fonte em Markdown: [`relatorio_final_artigo.md`](relatorio_final_artigo.md)

---

## Autores — Grupo nº 6

- **Estanislau da Fonseca**
- **Maria da Conceição**
- **Maurício Piquito**
- **Ulundo Aurélio Moisés**

*Mestrado em Engenharia Informática — Universidade Katyavala Bwila (2025/2026)*
