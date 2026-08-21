# -*- coding: utf-8 -*-
"""
UC: Aprendizagem Profunda - Trabalho Prático
Tarefa 3: Modelos de Deep Learning em PyTorch
Classificação Binária de Texto: Human (0) vs AI (1)

Modelos Implementados:
1. PyTorch Tabular DNN (com features TF-IDF)
2. PyTorch BiLSTM (com camada de Embeddings treinada de raiz)
3. PyTorch Transformer Encoder (com Positional Encoding e Multi-Head Self-Attention)
"""

import os
import math
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# =====================================================================
# 0. CONFIGURAÇÃO DE AMBIENTE E REPRODUTIBILIDADE
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Dispositivo de execução selecionado: {device}")

# Criar pasta para guardar os modelos treinados
os.makedirs("models", exist_ok=True)


# =====================================================================
# 1. CARREGAMENTO DOS DADOS LIMPOS (Gerados na Tarefa 1)
# =====================================================================
print("\n--- 1. Carregamento dos Datasets ---")
path_train = os.path.join("data", "train_clean.csv")
path_val = os.path.join("data", "val_clean.csv")
path_test = os.path.join("data", "test_clean.csv")

if not (os.path.exists(path_train) and os.path.exists(path_val) and os.path.exists(path_test)):
    raise FileNotFoundError("Datasets limpos não encontrados em 'data/'. Por favor execute tarefa1_construcao_data.py primeiro.")

df_train = pd.read_csv(path_train)
df_val = pd.read_csv(path_val)
df_test = pd.read_csv(path_test)

train_texts = df_train['text'].astype(str).tolist()
train_labels = df_train['label_num'].values.astype(np.float32)

val_texts = df_val['text'].astype(str).tolist()
val_labels = df_val['label_num'].values.astype(np.float32)

test_texts = df_test['text'].astype(str).tolist()
test_labels = df_test['label_num'].values.astype(np.float32)

print(f"Exemplos - Treino: {len(train_texts)} | Validação: {len(val_texts)} | Teste: {len(test_texts)}")
print(f"Distribuição de Classes no Treino: Humanos (0) = {np.sum(train_labels == 0)}, IA (1) = {np.sum(train_labels == 1)}")


# =====================================================================
# 2. DEFINIÇÃO DOS DATASETS E PIPELINES DE PROCESSAMENTO
# =====================================================================

# 2.1 Dataset Tabular (TF-IDF) para DNN
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 2.2 Tokenizador e Vocabulário Simples para Modelos Sequenciais
class SimpleTokenizer:
    def __init__(self, max_vocab_size=10000, max_seq_len=150):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        
    def fit(self, texts):
        token_counter = Counter()
        for text in texts:
            tokens = text.lower().strip().split()
            token_counter.update(tokens)
            
        most_common = token_counter.most_common(self.max_vocab_size - 2)
        for token, _ in most_common:
            self.vocab[token] = len(self.vocab)
            
    def encode(self, text):
        tokens = text.lower().strip().split()
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # Truncar ou adicionar Padding até max_seq_len
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices = indices + [self.vocab["<PAD>"]] * (self.max_seq_len - len(indices))
        return indices

    @property
    def vocab_size(self):
        return len(self.vocab)


# 2.3 Dataset Sequencial (para LSTM e Transformer)
class TextSequenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encoded_texts = [tokenizer.encode(t) for t in texts]
        self.X = torch.tensor(self.encoded_texts, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# 3. ARQUITETURAS DOS MODELOS EM PYTORCH
# =====================================================================

# 3.1 Modelo 1: Tabular Deep Neural Network (DNN)
class TabularDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3):
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
            
            nn.Linear(hidden_dim2, 1)  # Saída em logits para BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.network(x)


# 3.2 Modelo 2: Bidirectional LSTM (BiLSTM com Embeddings de Raiz)
class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_layers=2, dropout_rate=0.3):
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
        
        # Camada densa de classificação final
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embed_dropout(self.embedding(x)) # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)               # (batch_size, seq_len, hidden_dim * 2)
        
        # Global Max Pooling ao longo do tempo (sequência)
        pooled, _ = torch.max(lstm_out, dim=1)           # (batch_size, hidden_dim * 2)
        logits = self.fc(pooled)
        return logits


# 3.3 Modelo 3: Transformer Encoder com Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout_rate=0.2):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Mascarar tokens de preenchimento (<PAD>)
        padding_mask = (x == 0) # (batch_size, seq_len)
        
        out = self.embedding(x) * math.sqrt(self.d_model)
        out = self.pos_encoder(out)
        encoded = self.transformer_encoder(out, src_key_padding_mask=padding_mask)
        
        # Mean Pooling ponderado ignorando o padding
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# 4. MOTOR GENÉRICO DE TREINO, VALIDAÇÃO E AVALIAÇÃO
# =====================================================================
def train_and_evaluate_model(
    model, 
    train_loader, 
    val_loader, 
    test_loader, 
    optimizer, 
    criterion, 
    epochs=40, 
    patience=8, 
    model_save_path="models/best_model.pt",
    model_name="Modelo"
):
    print(f"\n=======================================================")
    print(f" Treino: {model_name}")
    print(f"=======================================================")
    
    model = model.to(device)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # --- Fase de Treino ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)
            
        train_loss /= total_train
        train_acc = correct_train / total_train
        
        # --- Fase de Validação ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == y_batch).sum().item()
                total_val += len(y_batch)
                
        val_loss /= total_val
        val_acc = correct_val / total_val
        
        # Mostrar progresso
        if epoch % 5 == 0 or epoch == 1:
            print(f"Época {epoch:02d}/{epochs} | Treino Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
            
        # Paragem Precoce & Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Paragem precoce acionada na época {epoch}! (Melhor Val Loss: {best_val_loss:.4f}, Acc: {best_val_acc*100:.2f}%)")
                break

    # --- Avaliação Final no Conjunto de Teste ---
    print(f"\n[*] A carregar o melhor checkpoint de '{model_save_path}' para teste...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(y_batch.flatten().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"\n--- Resultados no Conjunto de Teste: {model_name} ---")
    print(f"Exatidão (Accuracy):  {acc * 100:.2f}%")
    print(f"Precisão (Precision): {prec * 100:.2f}%")
    print(f"Revocação (Recall):   {rec * 100:.2f}%")
    print(f"F1-Score:             {f1 * 100:.2f}%")
    print("Matriz de Confusão:")
    print(cm)
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Checkpoint": model_save_path
    }


# =====================================================================
# 5. EXECUÇÃO PRINCIPAL DOS MODELOS
# =====================================================================
def main():
    criterion = nn.BCEWithLogitsLoss()
    results = []
    batch_size = 16

    # -------------------------------------------------------------
    # 5.1 Treinar Modelo 1: Tabular DNN (TF-IDF)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# MODELO 1: Deep Neural Network (DNN) com TF-IDF")
    print("#" * 60)
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_texts).toarray()
    X_val_tfidf = vectorizer.transform(val_texts).toarray()
    X_test_tfidf = vectorizer.transform(test_texts).toarray()
    
    train_tab_loader = DataLoader(TabularDataset(X_train_tfidf, train_labels), batch_size=batch_size, shuffle=True)
    val_tab_loader = DataLoader(TabularDataset(X_val_tfidf, val_labels), batch_size=batch_size, shuffle=False)
    test_tab_loader = DataLoader(TabularDataset(X_test_tfidf, test_labels), batch_size=batch_size, shuffle=False)
    
    dnn_model = TabularDNN(input_dim=1000, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3)
    dnn_optimizer = optim.AdamW(dnn_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    res_dnn = train_and_evaluate_model(
        dnn_model, train_tab_loader, val_tab_loader, test_tab_loader,
        dnn_optimizer, criterion, epochs=40, patience=8,
        model_save_path="models/best_pytorch_dnn.pt", model_name="PyTorch Tabular DNN"
    )
    results.append(res_dnn)

    # Guardar Vetorizador TF-IDF
    import pickle
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # -------------------------------------------------------------
    # 5.2 Preparação do Tokenizador para Modelos Sequenciais
    # -------------------------------------------------------------
    tokenizer = SimpleTokenizer(max_vocab_size=5000, max_seq_len=150)
    tokenizer.fit(train_texts)
    print(f"\n[*] Vocabulário construído com {tokenizer.vocab_size} tokens únicos.")
    
    with open("models/simple_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    train_seq_loader = DataLoader(TextSequenceDataset(train_texts, train_labels, tokenizer), batch_size=batch_size, shuffle=True)
    val_seq_loader = DataLoader(TextSequenceDataset(val_texts, val_labels, tokenizer), batch_size=batch_size, shuffle=False)
    test_seq_loader = DataLoader(TextSequenceDataset(test_texts, test_labels, tokenizer), batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------
    # 5.3 Treinar Modelo 2: BiLSTM com Embeddings de Raiz
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# MODELO 2: BiLSTM com Embeddings de Raiz")
    print("#" * 60)
    
    bilstm_model = TextBiLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=64,
        num_layers=2,
        dropout_rate=0.3
    )
    bilstm_optimizer = optim.Adam(bilstm_model.parameters(), lr=0.002)
    
    res_bilstm = train_and_evaluate_model(
        bilstm_model, train_seq_loader, val_seq_loader, test_seq_loader,
        bilstm_optimizer, criterion, epochs=40, patience=8,
        model_save_path="models/best_pytorch_bilstm.pt", model_name="PyTorch BiLSTM"
    )
    results.append(res_bilstm)

    # -------------------------------------------------------------
    # 5.4 Treinar Modelo 3: Transformer Encoder
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# MODELO 3: Transformer Encoder com Multi-Head Attention")
    print("#" * 60)
    
    transformer_model = TextTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout_rate=0.2
    )
    transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    res_transformer = train_and_evaluate_model(
        transformer_model, train_seq_loader, val_seq_loader, test_seq_loader,
        transformer_optimizer, criterion, epochs=40, patience=8,
        model_save_path="models/best_pytorch_transformer.pt", model_name="PyTorch Transformer"
    )
    results.append(res_transformer)

    # -------------------------------------------------------------
    # 6. TABELA COMPARATIVA FINAL
    # -------------------------------------------------------------
    df_results = pd.DataFrame(results)
    df_results['Accuracy'] = df_results['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    df_results['Precision'] = df_results['Precision'].apply(lambda x: f"{x*100:.2f}%")
    df_results['Recall'] = df_results['Recall'].apply(lambda x: f"{x*100:.2f}%")
    df_results['F1-Score'] = df_results['F1-Score'].apply(lambda x: f"{x*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("                 RESUMO COMPARATIVO DOS MODELOS (TAREFA 3 - PYTORCH)")
    print("=" * 80)
    print(df_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Checkpoint']].to_string(index=False))
    print("=" * 80)

    # -------------------------------------------------------------
    # 7. DEMONSTRAÇÃO DE INFERÊNCIA COM NOVO TEXTO
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("           DEMONSTRAÇÃO DE INFERÊNCIA EM TEMPO REAL (EXEMPLO)")
    print("=" * 80)
    
    amostras = [
        "In this study, we investigated the thermodynamic properties of superconducting materials at low temperatures.",
        "As an artificial intelligence language model developed by OpenAI, I can provide information and assist with tasks."
    ]
    
    bilstm_model.load_state_dict(torch.load("models/best_pytorch_bilstm.pt", map_location=device))
    bilstm_model.eval()
    
    with torch.no_grad():
        for texto in amostras:
            indices = torch.tensor([tokenizer.encode(texto)], dtype=torch.long).to(device)
            prob = torch.sigmoid(bilstm_model(indices)).item()
            classe = "IA (1)" if prob >= 0.5 else "Humano (0)"
            print(f"Texto: \"{texto[:75]}...\"")
            print(f"-> Predição: {classe} | Probabilidade de IA: {prob*100:.2f}%\n")


if __name__ == "__main__":
    main()
