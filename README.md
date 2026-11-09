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

## Submissão Oficial (Pasta `Submissao1/`)

Em estrito cumprimento das instruções da unidade curricular, foram classificados os 100 exemplos de teste disponibilizados em `dataset3_inputs.csv` através dos dois melhores modelos apurados no benchmark do **Grupo 6**.

Os resultados e o código utilizado encontram-se na pasta [`Submissao1/`](Submissao1/):

1. **1ª Submissão — Modelo 1 (s1):**
   - **Ficheiro:** [`Submissao1/submissao1-grupo6-s1.csv`](Submissao1/submissao1-grupo6-s1.csv)
   - **Modelo:** *PyTorch BiLSTM com Embeddings* (Rank #1 no Benchmark Geral)
   - **Métricas no Conjunto de Teste:** $F_1\text{-Score} = 66.67\%$, $\text{ROC-AUC} = 0.8333$, $\text{Acurácia} = 69.23\%$
   - **Coluna de Saídas:** `Output` com rótulos estritos `"Human"` ou `"AI"`.

2. **2ª Submissão — Modelo 2 (s2):**
   - **Ficheiro:** [`Submissao1/submissao1-grupo6-s2.csv`](Submissao1/submissao1-grupo6-s2.csv)
   - **Modelo:** *PyTorch Tabular DNN com TF-IDF* (Rank #2 no Benchmark Geral)
   - **Métricas no Conjunto de Teste:** $F_1\text{-Score} = 62.50\%$, $\text{ROC-AUC} = 0.8095$, $\text{Acurácia} = 53.85\%$
   - **Coluna de Saídas:** `Output` com rótulos estritos `"Human"` ou `"AI"`.

3. **Código de Geração e Reprodutibilidade:**
   - **Script Python:** [`Submissao1/gerar_submissao1.py`](Submissao1/gerar_submissao1.py) (script autocontido para regeneração direta via terminal)
   - **Jupyter Notebook:** [`Submissao1/submissao1_grupo6.ipynb`](Submissao1/submissao1_grupo6.ipynb) (inspeção interativa das predições)

---

## Estrutura do Repositório

```text
nosso_trabalho_ap_v_final_v2/
├── README.md                          # Documentação principal do projeto
├── relatorio_final_artigo.pdf         # Relatório científico final formatado (Artigo completo)
├── relatorio_final_artigo.md          # Versão integral em Markdown do artigo científico
│
├── Submissao1/                        # Pasta oficial da Submissão 1 do Grupo 6
│   ├── submissao1-grupo6-s1.csv       # Previsões oficiais do Modelo 1 (PyTorch BiLSTM)
│   ├── submissao1-grupo6-s2.csv       # Previsões oficiais do Modelo 2 (PyTorch Tabular DNN)
│   ├── gerar_submissao1.py            # Programa Python usado para gerar as previsões
│   └── submissao1_grupo6.ipynb        # Notebook interativo de execução da submissão
│
├── data/                              # Datasets brutos, conjuntos limpos e dados de validação
│   ├── dataset1_inputs.csv            # Textos brutos do primeiro lote de dados
│   ├── dataset1_outputs.csv           # Rótulos de referência (Human vs AI) do primeiro lote
│   ├── dataset2_complete.csv          # Segundo lote de dados consolidado
│   ├── dataset3_inputs.csv            # Conjunto de 100 exemplos para a Submissão 1
│   ├── train_clean.csv                # Conjunto de treino limpo e estratificado (80%)
│   ├── val_clean.csv                  # Conjunto de validação balanceado (10%)
│   ├── test_clean.csv                 # Conjunto de teste independente não-visto (10%)
│   └── simulated_teacher_validation.csv # Amostra para validação do pipeline do docente
│
├── models/                            # Pesos e pré-processadores serializados
│   ├── best_pytorch_bilstm.pt         # Pesos da Bidirectional LSTM (PyTorch) — TOP #1 Global
│   ├── best_pytorch_dnn.pt            # Pesos da Tabular DNN TF-IDF (PyTorch) — TOP #2 Global
│   ├── best_pytorch_transformer.pt    # Pesos do Transformer Encoder (PyTorch)
│   ├── best_numpy_lr.pkl              # Pesos da Regressão Logística (NumPy de raiz)
│   ├── best_numpy_dnn.pkl             # Pesos da Rede Neuronal Profunda / MLP (NumPy de raiz)
│   ├── tfidf_vectorizer.pkl           # Vetorizador TF-IDF ajustado
│   └── simple_tokenizer.pkl           # Tokenizador e vocabulário sequencial
│
├── notebooks/                         # Notebooks de experimentação, inspeção e submissão
│   ├── tarefa4_submissao_docente.ipynb# Notebook principal de inferência do docente
│   ├── tarefa3_pytorch.ipynb          # Treino e ajuste de hiperparâmetros em PyTorch
│   └── inspecionar_modelos.ipynb      # Inspeção detalhada de pesos e vocabulários
│
├── results/                           # Gráficos de benchmark, curvas ROC e matrizes de confusão
│   ├── benchmark_ranking_table.csv    # Tabela consolidada com todas as métricas de avaliação
│   ├── benchmark_metrics_comparison.png # Gráfico de barras comparativo de desempenho
│   ├── roc_curves_all_models.png      # Curvas ROC e áreas sob a curva (ROC-AUC)
│   ├── confusion_matrices_all_models.png # Matrizes de confusão dos 5 modelos
│   └── latency_vs_accuracy.png        # Análise de trade-off entre Latência (ms) e F1-Score
│
├── src/                               # Módulos Python (Framework de raiz e Inferência)
│   ├── activation.py                  # Sigmoid, ReLU, Tanh, Softmax e respetivas derivadas
│   ├── data.py                        # Carregamento, batching e pré-processamento de datasets
│   ├── layers.py                      # Camada DenseLayer com forward e backward pass manuais
│   ├── losses.py                      # Binary Cross-Entropy, MSE e respetivos gradientes
│   ├── metrics.py                     # Cálculo manual de Accuracy, Precision, Recall e F1
│   ├── neuralnetwork.py               # Classe NeuralNetwork em NumPy puro
│   ├── optimizer.py                   # Otimizadores manuais (SGD com Momentum e Adam)
│   └── inference.py                   # Motor unificado ModelInferenceEngine para inferência
│
├── tarefa1_construcao_data.py         # Tarefa 1: Curadoria, limpeza e divisão estratificada
├── tarefa2_numpay.py                  # Tarefa 2: Treino e avaliação dos modelos de raiz em NumPy
├── tarefa3_pytorch.py                 # Tarefa 3: Treino dos modelos Deep Learning em PyTorch
├── tarefa4_avaliacao_modelos.py       # Tarefa 4: Benchmark independente e curvas comparativas
└── inspecionar_modelos.py             # Script de diagnóstico e inspeção de artefactos
```

---

## Tabela de Benchmark Comparativo (Resultados Obtidos)

Avaliação realizada no conjunto de teste independente (`data/test_clean.csv`):

| Rank | Modelo | Framework / Abordagem | Exatidão | F1-Score | Precisão | Recall | ROC-AUC | Latência / Amostra |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **PyTorch BiLSTM (Embeddings)** | PyTorch | **69.23%** | **66.67%** | **66.67%** | **66.67%** | **0.8333** | 8.96 ms |
| **#2** | **PyTorch Tabular DNN (TF-IDF)** | PyTorch | 53.85% | 62.50% | 50.00% | **83.33%** | 0.8095 | 12.82 ms |
| **#3** | **PyTorch Transformer Encoder** | PyTorch | 38.46% | 50.00% | 40.00% | 66.67% | 0.6190 | 8.97 ms |
| **#4** | **NumPy Logistic Reg (Baseline)** | NumPy Puro (de raiz) | 46.15% | 36.36% | 40.00% | 33.33% | 0.4286 | 2.43 ms |
| **#5** | **NumPy DNN / MLP** | NumPy Puro (de raiz) | 38.46% | 33.33% | 33.33% | 33.33% | 0.4286 | **1.57 ms** |

---

## Como Reproduzir a Submissão 1

Para gerar ou verificar os ficheiros oficiais de submissão através da consola:

```bash
python Submissao1/gerar_submissao1.py
```

O script carregará automaticamente os pesos pré-treinados em `models/`, efetuará o pré-processamento dos textos de `dataset3_inputs.csv` e gravará os dois ficheiros CSV (`submissao1-grupo6-s1.csv` e `submissao1-grupo6-s2.csv`) com as predições formatadas estritamente com os rótulos `"Human"` e `"AI"`.

Em alternativa, o notebook interativo [`Submissao1/submissao1_grupo6.ipynb`](Submissao1/submissao1_grupo6.ipynb) pode ser executado célula a célula em ambiente Jupyter ou Google Colab.

---

## Como Executar as Tarefas do Projeto Localmente

### 1. Pré-requisitos e Dependências

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
