# -*- coding: utf-8 -*-
"""
=============================================================================
Universidade Katyavala Bwila - Instituto Politécnico
Mestrado em Engenharia Informática - Aprendizagem Profunda 2025/2026
Docentes: Prof. Doutor Miguel Rocha & Prof. Doutor Lázaro Makili

Submissão 1 - Grupo de Trabalho nº 6
- Estanislau da Fonseca
- Maria da Conceição
- Maurício Piquito
- Ulundo Aurélio Moisés

Objetivo:
Classificação dos 100 textos não rotulados de 'dataset3_inputs.csv' utilizando
os dois melhores modelos apurados no benchmark do projeto:
  1. Submissão 1 (s1): PyTorch BiLSTM com Embeddings (Rank #1 - F1: 66.67%, ROC-AUC: 0.8333)
  2. Submissão 2 (s2): PyTorch Tabular DNN com TF-IDF (Rank #2 - F1: 62.50%, ROC-AUC: 0.8095)

Ficheiros de saída gerados (conforme instrução oficial):
  - Submissao1/submissao1-grupo6-s1.csv
  - Submissao1/submissao1-grupo6-s2.csv
=============================================================================
"""

import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn

# Configurar diretórios base
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUBMISSAO_DIR = CURRENT_DIR

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# =============================================================================
# 1. DEFINIÇÕES DAS ARQUITETURAS E PRÉ-PROCESSADORES
# =============================================================================
class SimpleTokenizer:
    def __init__(self, max_vocab_size=10000, max_seq_len=150):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}

    def encode(self, text):
        tokens = str(text).lower().strip().split()
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.vocab["<PAD>"]] * (self.max_seq_len - len(indices))
        return indices

    @property
    def vocab_size(self):
        return len(self.vocab)


class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(TextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embedded = self.embed_dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled, _ = torch.max(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits


class TabularDNN(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3):
        super(TabularDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        return self.network(x)


class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SimpleTokenizer':
            return SimpleTokenizer
        if module == '__main__':
            if name in globals():
                return globals()[name]
        return super().find_class(module, name)


def carregar_pickle(caminho):
    with open(caminho, "rb") as f:
        return RobustUnpickler(f).load()


# =============================================================================
# 2. LOCALIZAR E CARREGAR DATASET3_INPUTS.CSV
# =============================================================================
def encontrar_dataset3():
    candidatos = [
        os.path.join(PROJECT_ROOT, "dataset3_inputs.csv"),
        os.path.join(DATA_DIR, "dataset3_inputs.csv"),
        os.path.join(CURRENT_DIR, "dataset3_inputs.csv")
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Não foi possível encontrar 'dataset3_inputs.csv' em nenhum dos caminhos: {candidatos}")


def carregar_dados_teste(caminho_csv):
    print(f"[*] A carregar ficheiro de entrada: {caminho_csv}")
    # Tratar encoding com BOM e delimitadores comuns (; ou ,)
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(caminho_csv, sep=sep, encoding="utf-8-sig")
            if len(df.columns) >= 2:
                break
        except Exception:
            continue

    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    
    # Identificar coluna ID
    id_col = None
    for c in df.columns:
        if c.lower() in ["id", "index", "sample_id"]:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    # Identificar coluna Text
    text_col = None
    for c in df.columns:
        if c.lower() in ["text", "input", "content", "body"]:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[1]

    ids = df[id_col].astype(str).tolist()
    texts = df[text_col].astype(str).tolist()
    
    print(f"[+] Total de exemplos carregados: {len(texts)}")
    print(f"    Coluna ID: '{id_col}' | Coluna Texto: '{text_col}'")
    return ids, texts


# =============================================================================
# 3. PIPELINE DE CLASSIFICAÇÃO
# =============================================================================
def inferir_bilstm(texts, models_dir, device):
    tok_path = os.path.join(models_dir, "simple_tokenizer.pkl")
    weights_path = os.path.join(models_dir, "best_pytorch_bilstm.pt")
    
    tokenizer = carregar_pickle(tok_path)
    model = TextBiLSTM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    encoded = [tokenizer.encode(t) for t in texts]
    x_tensor = torch.tensor(encoded, dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    labels = ["AI" if p >= 0.5 else "Human" for p in probs]
    return labels, probs


def inferir_tabular_dnn(texts, models_dir, device):
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    weights_path = os.path.join(models_dir, "best_pytorch_dnn.pt")
    
    vectorizer = carregar_pickle(vec_path)
    input_dim = len(vectorizer.get_feature_names_out())
    model = TabularDNN(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    x_tfidf = vectorizer.transform(texts).toarray()
    x_tensor = torch.tensor(x_tfidf, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    labels = ["AI" if p >= 0.5 else "Human" for p in probs]
    return labels, probs


# =============================================================================
# 4. EXECUÇÃO PRINCIPAL
# =============================================================================
def main():
    print("=" * 78)
    print("  GERADOR OFICIAL DE SUBMISSÕES 1 - APRENDIZAGEM PROFUNDA (GRUPO 6)")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de inferência: {device}")

    # 1. Carregar os 100 exemplos
    caminho_dataset = encontrar_dataset3()
    ids, texts = carregar_dados_teste(caminho_dataset)

    # 2. Executar Modelo 1 (s1) - PyTorch BiLSTM
    print("\n" + "-" * 78)
    print("[*] 1ª SUBMISSÃO: PyTorch BiLSTM (Embeddings) -> s1")
    print("-" * 78)
    labels_s1, probs_s1 = inferir_bilstm(texts, MODELS_DIR, device)
    
    df_s1 = pd.DataFrame({
        "ID": ids,
        "Output": labels_s1
    })
    
    out_s1_path = os.path.join(SUBMISSAO_DIR, "submissao1-grupo6-s1.csv")
    df_s1.to_csv(out_s1_path, index=False)
    print(f"[+] Ficheiro gerado com sucesso: {out_s1_path}")
    print(f"    Contagem de labels: Human = {sum(1 for l in labels_s1 if l == 'Human')}, AI = {sum(1 for l in labels_s1 if l == 'AI')}")

    # 3. Executar Modelo 2 (s2) - PyTorch Tabular DNN
    print("\n" + "-" * 78)
    print("[*] 2ª SUBMISSÃO: PyTorch Tabular DNN (TF-IDF) -> s2")
    print("-" * 78)
    labels_s2, probs_s2 = inferir_tabular_dnn(texts, MODELS_DIR, device)
    
    df_s2 = pd.DataFrame({
        "ID": ids,
        "Output": labels_s2
    })
    
    out_s2_path = os.path.join(SUBMISSAO_DIR, "submissao1-grupo6-s2.csv")
    df_s2.to_csv(out_s2_path, index=False)
    print(f"[+] Ficheiro gerado com sucesso: {out_s2_path}")
    print(f"    Contagem de labels: Human = {sum(1 for l in labels_s2 if l == 'Human')}, AI = {sum(1 for l in labels_s2 if l == 'AI')}")

    # 4. Resumo e Concordância entre os Modelos
    print("\n" + "=" * 78)
    print("  RESUMO E VALIDAÇÃO DOS RESULTADOS GERADOS")
    print("=" * 78)
    concordancias = sum(1 for l1, l2 in zip(labels_s1, labels_s2) if l1 == l2)
    print(f"[*] Exemplos analisados: {len(ids)}")
    print(f"[*] Ficheiro s1: {out_s1_path} ({os.path.getsize(out_s1_path)} bytes)")
    print(f"[*] Ficheiro s2: {out_s2_path} ({os.path.getsize(out_s2_path)} bytes)")
    print(f"[*] Taxa de concordância entre os dois modelos: {concordancias}/{len(ids)} ({concordancias/len(ids)*100:.1f}%)")
    print("[+] Concluído conforme estritamente estipulado em 'instrucoes.pdf'.\n")


if __name__ == "__main__":
    main()

