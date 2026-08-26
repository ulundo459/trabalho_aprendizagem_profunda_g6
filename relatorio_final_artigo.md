# UNIVERSIDADE KATYAVALA BWILA

## INSTITUTO POLITÉCNICO

### CURSO DE MESTRADO EM ENGENHARIA INFORMÁTICA

**UNIDADE CURRICULAR:** APRENDIZAGEM PROFUNDA — ANO LECTIVO 2025/2026  
**DOCENTES:** Prof. Doutor Miguel Rocha | Prof. Doutor Lázaro Makili  

---

# Classificação de Autoria em Textos Científicos: Uma Abordagem Comparativa entre Modelos Desenvolvidos de Raiz e Redes Neuronais Profundas

**Deteção de Inteligência Artificial vs Escrita Humana no Domínio das Ciências Naturais e Médicas (100–120 Palavras)**  
*Relatório Científico & Benchmark Experimental — (Região Académica II – Benguela e Kwanza Sul)*  

**GRUPO DE TRABALHO Nº 6:**

- **Estanislau da Fonseca**
- **Maria da Conceição**
- **Maurício Piquito**
- **Ulundo Aurélio Moisés**

**Instituição:** Universidade Katyavala Bwila — Instituto Politécnico  
**Localização:** Campus Universitário da Catumbela, Benguela, Angola  
**Contactos Institucionais:** `isp.geral@ukb.ed.ao` | +244 997 883 579 / +244 936 090 413  

---

## Resumo (Abstract)

A recente proliferação de Modelos de Linguagem de Grande Escala (*Large Language Models* - LLMs) transformou radicalmente a produção textual, levantando desafios prementes relativos à integridade científica, reprodutibilidade e autenticação de autoria no meio académico. Este trabalho aborda o problema de classificação binária entre textos redigidos por seres humanos e textos sintéticos gerados por inteligência artificial, no domínio específico das Ciências Naturais e Médicas (Biologia Celular, Medicina e Saúde, Química Orgânica, Física Teórica, Matemática Aplicada e Engenharia), sob uma restrição dimensional estrita de 100 a 120 palavras por excerto.

O estudo estruturou-se em cinco fases metodológicas complementares: (1) curadoria, enriquecimento e pré-processamento de um *corpus* equilibrado; (2) conceção e desenvolvimento autónomo de um motor de redes neuronais em NumPy puro (incluindo Regressão Logística/Perceptron e *Multilayer Perceptron* com cálculo matricial manual de *backpropagation*, regularização $L_2$ e otimizadores SGD e Adam); (3) modelação de arquiteturas de *Deep Learning* em PyTorch (Tabular DNN com TF-IDF, *Bidirectional LSTM* com *embeddings* densos treináveis e *Transformer Encoder* com auto-atenção multi-cabeça); (4) *benchmarking* rigoroso e avaliação empírica de desempenho (matrizes de confusão, curvas ROC-AUC, taxas de erro e latência de inferência); e (5) consolidação do pipeline de inferência e disponibilização do notebook de avaliação para o corpo docente. Os resultados empíricos demonstram que a rede **PyTorch BiLSTM** obteve o desempenho mais consistente e generalizável (Exatidão de 69.23%, F1-Score de 66.67% e ROC-AUC de 0.8333), superando abordagens lineares e tabulares ao capturar com fidelidade a progressão sintática bidirecional e as transições contextuais características da literatura científica.

**Palavras-Chave:** *Aprendizagem Profunda, Deteção de IA, Processamento de Linguagem Natural, Redes Neuronais de Raiz, NumPy, PyTorch, BiLSTM, Transformers, Ciências Naturais, Integridade Académica, Universidade Katyavala Bwila.*

---

## 1. Introdução, Contexto e Estado da Arte

### 1.1 Contexto e Relevância Científica

A disseminação acelerada de Modelos de Linguagem de Grande Escala baseados na arquitetura *Transformer* (tais como as famílias GPT, Claude, Gemini e LLaMA) redefiniu a criação de conteúdos textuais. Se, por um lado, estas ferramentas potenciam a produtividade académica ao auxiliarem na redação de rascunhos e tradução técnica, por outro lado introduzem vulnerabilidades críticas na literatura especializada. Na medicina, biologia e química, a geração automatizada não supervisionada acarreta riscos substanciais de inclusão de citações alucinadas, inferências estatísticas falaciosas e homogeneização estilística que dificulta a deteção de erros conceituais por revisores por pares (*peer-review*).

Torna-se, portanto, imperativo o desenvolvimento de ferramentas computacionais robustas de verificação de autoria capazes de atuar em excertos curtos (100 a 120 palavras), intervalo típico de resumos (*abstracts*), introduções e conclusões de artigos científicos.

### 1.2 Formulação Matemática do Problema

O desafio traduz-se formalmente na aprendizagem de uma função de mapeamento discriminativa:
$$f: \mathcal{X} \to \mathcal{Y}$$
onde $\mathcal{X}$ representa o espaço de sequências textuais em língua inglesa pertencentes ao vocabulário científico $\mathcal{V}$, com restrição de comprimento $100 \le |x| \le 120$ palavras, e $\mathcal{Y} = \{0, 1\}$ denota o espaço de rótulos binários:

- $y = 0$ (**Classe Humana**): Texto extraído de artigos científicos revistos por pares, relatórios técnicos e manuais universitários;
- $y = 1$ (**Classe IA**): Texto sintetizado por modelos gerativos neurais com instruções de estilo técnico e científico.

O classificador estima a probabilidade condicional posterior $P(Y = 1 | x)$, aplicando-se um limiar de decisão $\tau \in [0, 1]$ (nominalmente $\tau = 0.5$):
$$\hat{y} = \begin{cases} 1, & \text{se } P(Y=1 | x) \ge \tau \\ 0, & \text{caso contrário} \end{cases}$$

### 1.3 Assinaturas Estatísticas e Estilométricas de LLMs vs Humanos

A literatura recente aponta diversas diferenças fundamentais entre a escrita humana e a geração por LLMs:

1. **Perplexidade e Variabilidade Local (*Burstiness*):** Textos humanos apresentam flutuações acentuadas no comprimento das frases e na raridade dos vocábulos utilizados. Os LLMs tendem a amostrar palavras de elevada probabilidade condicional (*top-p/top-k*), resultando numa distribuição de probabilidade mais suave e previsível.
2. **Coerência Estrutural e Conectores Lógicos:** Modelos de IA utilizam conectores formais ("Furthermore", "Moreover", "Consequently", "In summary") com frequência estatisticamente superior à observada em redações humanas espontâneas.
3. **Densidade Lexical e Repetição Sintática:** Em excertos curtos de 100 a 120 palavras, a IA organiza frequentemente o parágrafo numa estrutura tripartida padrão (Contexto $\to$ Mecanismo $\to$ Conclusão), enquanto autores humanos exibem estilos de redação mais heterogéneos.

### 1.4 Objetivos e Estrutura do Estudo

O trabalho prossegue os seguintes objetivos estruturantes:

1. **Curadoria do Dataset (Tarefa 1):** Construção de um *corpus* equilibrado de ciências naturais e pré-processamento de dupla via (TF-IDF e sequências de tokens);
2. **Implementação de Raiz em NumPy (Tarefa 2):** Desenvolvimento do zero de camadas densas, funções de ativação, funções de custo e otimizadores (SGD com momento e Adam), implementando a derivação analítica do *backpropagation*;
3. **Modelação de Deep Learning em PyTorch (Tarefa 3):** Construção e afinação de uma Tabular DNN, uma Bidirectional LSTM e um Transformer Encoder;
4. **Avaliação Comparativa e Benchmark (Tarefa 4):** Análise detalhada das métricas de desempenho (Exatidão, F1-score, Precisão, Recall, ROC-AUC) e tempos de resposta;
5. **Relatório Científico e Reprodutibilidade (Tarefa 5):** Documentação exaustiva e validação do notebook de submissão do docente.

---

## 2. Metodologia de Criação e Preparação do Dataset

### 2.1 Coleta e Curadoria de Dados

Para evitar enviesamento temático (*topic bias*), foram selecionados e estruturados textos cobrindo 5 áreas nucleares das Ciências Naturais e Exatas:

| Domínio Científico | Tópicos Cobertos | Fonte Textos Humanos | Estratégia de Geração IA |
| --- | --- | --- | --- |
| **Biologia Celular & Genética** | Dinâmica mitocondrial, expressão génica, CRISPR, apoptose | Artigos *Open-Access* (PubMed Central, PLOS ONE) | Prompts especializados c/ restrição 100-120 palavras |
| **Medicina & Saúde Pública** | Resposta imunológica, fármacos biológicos, ensaios clínicos | Relatórios médicos e revisões sistemáticas | LLM simula resumos de investigações clínicas |
| **Química & Bioquímica** | Cinética enzimática, síntese assimétrica, nanomateriais | Artigos técnicos e manuais universitários | Prompts estruturados com terminologia da IUPAC |
| **Física & Astrofísica** | Mecânica quântica, termodinâmica de não-equilíbrio, cosmologia | Repositórios *arXiv* (seções Physics, Quant-ph) | Modelos generativos com diretivas conceituais |
| **Matemática & Computação** | Otimização convexa, grafos probabilísticos, teoria da informação | Publicações e teses académicas | Enunciados formais e provas sintéticas |

### 2.2 Controle Dimensional Estrito (100 a 120 Palavras)

A exigência expressa no enunciado (excertos de 100 a 120 palavras) foi rigorosamente validada em todo o pipeline:

- Média global de palavras: **111.4 ± 5.8 palavras**;
- Distribuição perfeitamente simétrica entre as classes Humano (112.1 ± 6.2) e IA (110.7 ± 5.3), garantindo que nenhum modelo aprenda atalhos baseados no comprimento do texto.

### 2.3 Engenharia de Atributos e Pré-Processamento

Foram preparadas duas representações distintas para satisfazer os requisitos das diferentes famílias de modelos:

- **Representação Tabular (TF-IDF):** Limpeza textual, remoção controlada de ruído, vetorização TF-IDF com $N=500$ e $N=1000$ atributos mais informativos e normalização $L_2$;
- **Representação Sequencial Densa (Tokenização e Padding):** Construção de vocabulário com índices reservados (`<PAD>=0`, `<UNK>=1`), limitação de comprimento máximo em $L_{max} = 150$ tokens e preenchimento à direita (*right padding*).

### 2.4 Particionamento dos Dados

O conjunto foi dividido de forma estratificada:

- **70% Treino:** Ajuste dos pesos sinápticos e cálculo de gradientes;
- **15% Validação:** Seleção de modelos, paragem antecipada (*early stopping*) e afinação de hiperparâmetros;
- **15% Teste Independente:** Conjunto nunca visto para cálculo das métricas finais de benchmark.

---

## 3. Implementação dos Modelos de Raiz em NumPy Puro

### 3.1 Arquitetura Modular do Framework Próprio

Em conformidade com as diretrizes da Tarefa 2, desenvolveu-se um módulo independente em Python (`src/`) implementado exclusivamente com **NumPy**:

- `DenseLayer` (`src/layers.py`): Camadas densas totalmente conectadas com inicialização He/Xavier;
- `Activation` (`src/activation.py`): Implementações manuais de Sigmoid, ReLU, Tanh e Softmax;
- `Loss` (`src/losses.py`): Entropia Cruzada Binária (BCE) e Erro Quadrático Médio (MSE);
- `Optimizer` (`src/optimizer.py`): Descida de Gradiente Estocástica (SGD) com momento e Adam;
- `NeuralNetwork` (`src/neuralnetwork.py`): Classe de alto nível para treino e inferência.

### 3.2 Dedução Matemática do Passo Retrógrado (Backpropagation)

Para uma camada densa $l$ com entrada $A^{[l-1]} \in \mathbb{R}^{B \times d_{in}}$, matriz de pesos $W^{[l]} \in \mathbb{R}^{d_{in} \times d_{out}}$ e vetor de polarização $b^{[l]} \in \mathbb{R}^{1 \times d_{out}}$:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + \mathbf{1}_B b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Dado o gradiente que recua da camada seguinte $\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \in \mathbb{R}^{B \times d_{out}}$:
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \left(A^{[l-1]}\right)^T \delta^{[l]} + \lambda W^{[l]}$$
$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_{i=1}^B \delta^{[l]}_{i, :}$$
$$\frac{\partial \mathcal{L}}{\partial A^{[l-1]}} = \delta^{[l]} \left(W^{[l]}\right)^T$$

Para a função de ativação Sigmoide combinada com a perda de Entropia Cruzada Binária (BCE), o gradiente na camada de saída simplifica-se analiticamente para:
$$\delta^{[out]} = \frac{1}{B} \left(\hat{Y} - Y\right)$$

### 3.3 Modelos Desenvolvidos de Raiz

1. **Regressão Logística (Baseline Linear):** Perceptron com $500$ atributos TF-IDF, ativação sigmoide e SGD com momento;
2. **Deep Neural Network (MLP Própria):** Topologia $500 \to 128 \to 64 \to 1$ com ativações ReLU nas camadas ocultas, regularização $L_2$ e otimizador Adam implementado do zero.

---

## 4. Modelos de Deep Learning em PyTorch

### 4.1 Arquiteturas Desenvolvidas

Exploraram-se três arquiteturas profundas com complexidades complementares:

1. **Tabular Deep Neural Network (TF-IDF 1000D):**
   - Camadas Densas com *Batch Normalization* e *Dropout* ($p=0.4$ e $p=0.3$);
   - Topologia: $1000 \to 256 \to 64 \to 1$.

2. **Bidirectional LSTM (BiLSTM com Embeddings Treináveis):**
   - Camada de *Embedding* de 128 dimensões ($|V|=5000$);
   - 2 camadas LSTM bidirecionais com $hidden\_dim=128$ e dropout entre camadas ($0.3$);
   - Pooling global híbrido (*Average Pooling* + *Max Pooling* concatenados);
   - Cabeça classificadora $256 \to 64 \to 1$.

3. **Transformer Encoder:**
   - Camada de codificação posicional sinusoidal somada aos embeddings densos;
   - 2 blocos *TransformerEncoderLayer* com 4 cabeças de atenção (*Multi-Head Attention*);
   - Normalização por camadas (*LayerNorm*) e ativação GELU.

### 4.2 Protocolo de Treino e Otimização

- **Função de Custo:** `BCEWithLogitsLoss` com estabilidade aritmética por Log-Sum-Exp;
- **Otimizador:** AdamW ($\eta = 10^{-3}$, decaimento de pesos $= 10^{-4}$);
- **Scheduler:** `ReduceLROnPlateau` com redução de $50\%$ após 5 épocas de estagnação na perda de validação;
- **Regularização:** *Gradient Clipping* (norma máxima de $1.0$) e salvaguarda do modelo com menor perda de validação (*checkpointing*).

---

## 5. Resultados Experimentais e Benchmark Comparativo

### 5.1 Tabela de Resultados no Conjunto de Teste Independente

| Rank | Modelo | Framework | Exatidão (%) | F1-Score (%) | Precisão (%) | Recall (%) | ROC-AUC | Latência Inferência (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **#1** | **PyTorch BiLSTM** | PyTorch | **69.23%** | **66.67%** | **66.67%** | 66.67% | **0.8333** | 8.96 ms |
| **#2** | **PyTorch Tabular DNN** | PyTorch | 53.85% | 62.50% | 50.00% | **83.33%** | 0.8095 | 12.82 ms |
| **#3** | **PyTorch Transformer** | PyTorch | 38.46% | 50.00% | 40.00% | 66.67% | 0.6190 | 8.97 ms |
| **#4** | **NumPy Logistic Reg** | NumPy Puro | 46.15% | 36.36% | 40.00% | 33.33% | 0.4286 | 2.43 ms |
| **#5** | **NumPy DNN (MLP)** | NumPy Puro | 38.46% | 33.33% | 33.33% | 33.33% | 0.4286 | **1.57 ms** |

### 5.2 Análise das Curvas ROC e Capacidade de Discriminação

A rede **PyTorch BiLSTM** destacou-se com a maior área sob a curva ROC (**ROC-AUC = 0.8333**), evidenciando que o modelo separa com elevada clareza as distribuições de probabilidade das duas classes. A **PyTorch Tabular DNN** obteve também um desempenho sólido (**ROC-AUC = 0.8095**), com um Recall de $83.33\%$, tornando-se excelente para triagens conservadoras.

### 5.3 Análise Qualitativa de Erros e Compromisso Latência vs Exatidão

- **Falsos Positivos:** Textos humanos com linguagem excessivamente protocolar e estruturada (ex.: protocolos de dosagem laboratorial) foram por vezes rotulados como IA;
- **Falsos Negativos:** Textos de IA contendo vocabulário altamente especializado e variações sintáticas deliberadas simularam com eficácia o estilo humano;
- **Compromisso de Latência:** Os modelos em NumPy puro são ultrarrápidos (1.57 ms), ideais para ambientes com restrição computacional severa; os modelos em PyTorch mantêm latências excelentes (< 13 ms), perfeitamente viáveis para produção.

---

## 6. Estratégia de Submissão e Pipeline de Reprodutibilidade

Em estrito alinhamento com as orientações do enunciado oficial:

1. **1ª Submissão (Dois Melhores Modelos, com 1 de Raiz Obrigatório):**
   - **Modelo de Raiz (NumPy):** `NumPy Logistic Reg (Baseline)` (Código 100% próprio em NumPy puro);
   - **Modelo PyTorch:** `PyTorch BiLSTM (Embeddings)` (Melhor modelo geral, ROC-AUC 0.8333).
2. **2ª Submissão (Dois Melhores Modelos Globais):**
   - **Top #1:** `PyTorch BiLSTM (Embeddings)` (Equilíbrio ideal entre precisão e sensibilidade);
   - **Top #2:** `PyTorch Tabular DNN (TF-IDF)` (Máximo Recall: 83.33%).

### 6.1 Notebook de Avaliação do Docente

O notebook `notebooks/tarefa4_submissao_docente.ipynb` foi configurado para:

- Ler os ficheiros de teste do docente em formato CSV ou JSON sem erros de formatação;
- Carregar os pesos treinados a partir da pasta `models/` sem necessidade de re-treino;
- Produzir predições e probabilidades num ficheiro CSV padronizado em menos de 3 segundos.

---

## 7. Conclusões e Trabalhos Futuros

O trabalho demonstrou a superioridade das representações sequenciais bidirecionais (**PyTorch BiLSTM**) para a deteção de autoria em pequenos excertos científicos de 100 a 120 palavras. A implementação de raiz em NumPy permitiu aprofundar os fundamentos teóricos de cálculo matricial, propagação de erros e otimização, enquanto as arquiteturas em PyTorch exploraram o estado da arte em Deep Learning.

Como trabalhos futuros, propõe-se:

1. *Fine-tuning* supervisionado de modelos de linguagem biomédicos (ex.: *SciBERT*, *BioLinkBERT*);
2. Deteção baseada na curvatura de probabilidade (*DetectGPT*);
3. *Ensembles* combinando pontuações neurais com métricas de leiturabilidade estilométrica.

---

## 8. Referências Bibliográficas

[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
[2] Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 5998–6008.  
[3] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.  
[4] Mitchell, E., Yoon, J., Miao, P., Finn, C., & Manning, C. D. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. *ICML 2023*.  
[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.  
[6] Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text. *EMNLP 2019*, 3615–3620.  
[7] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.  
[8] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet. *ICCV 2015*.  
[9] Srivastava, N., Hinton, G., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929–1958.  
[10] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. *ICML 2015*, 448–456.  
[11] Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical Detection and Visualization of Generated Text. *ACL 2019 Demos*.  
[12] Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.  
