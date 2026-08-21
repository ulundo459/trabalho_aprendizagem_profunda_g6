# -*- coding: utf-8 -*-
"""
Módulo de Inferência e Submissão Unificada
UC: Aprendizagem Profunda - Trabalho Prático

Suporta carregamento, pré-processamento e inferência para todos os 5 modelos:
1. numpy_lr (Regressão Logística NumPy de Raiz)
2. numpy_dnn (Rede Neuronal Profunda NumPy de Raiz)
3. pytorch_dnn (PyTorch Tabular DNN com TF-IDF)
4. pytorch_bilstm (PyTorch BiLSTM com Embeddings)
5. pytorch_transformer (PyTorch Transformer Encoder)
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

# Garantir importação do módulo src
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import Data
from layers import DenseLayer
from activation import SigmoidActivation, ReLUActivation
from losses import BinaryCrossEntropy
from optimizer import Optimizer
from neuralnetwork import NeuralNetwork
from metrics import accuracy


# =====================================================================
# 1. DEFINIÇÃO DO TOKENIZADOR
# =====================================================================
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


# =====================================================================
# 2. DEFINIÇÃO DAS ARQUITETURAS PYTORCH
# =====================================================================
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TextTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout_rate=0.2):
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
        padding_mask = (x == 0)
        out = self.embedding(x) * math.sqrt(self.d_model)
        out = self.pos_encoder(out)
        encoded = self.transformer_encoder(out, src_key_padding_mask=padding_mask)
        
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        logits = self.classifier(pooled)
        return logits


class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SimpleTokenizer':
            return SimpleTokenizer
        if module == '__main__':
            # Se a classe estiver em globals ou no módulo local
            if name in globals():
                return globals()[name]
        return super().find_class(module, name)

def robust_pickle_load(file_path):
    with open(file_path, "rb") as f:
        return RobustUnpickler(f).load()

# =====================================================================
# 3. GESTOR CENTRAL DE INFERÊNCIA E MODELOS
# =====================================================================
class ModelInferenceEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = None
        self.tokenizer = None
        self.cached_models = {}
        
        self._load_preprocessors()
        
    def _load_preprocessors(self):
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        tok_path = os.path.join(self.models_dir, "simple_tokenizer.pkl")
        
        if os.path.exists(vec_path):
            self.vectorizer = robust_pickle_load(vec_path)
                
        if os.path.exists(tok_path):
            self.tokenizer = robust_pickle_load(tok_path)

    def get_model(self, model_type="pytorch_bilstm"):
        """Carrega e instancia o modelo solicitado a partir do disco."""
        if model_type in self.cached_models:
            return self.cached_models[model_type]
            
        model_type_lower = model_type.lower().strip()
        
        if model_type_lower in ["numpy_lr", "numpy_logistic"]:
            path = os.path.join(self.models_dir, "best_numpy_lr.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modelo {path} não encontrado.")
            model = robust_pickle_load(path)
            self.cached_models[model_type] = model
            return model
            
        elif model_type_lower in ["numpy_dnn", "numpy_neural_net"]:
            path = os.path.join(self.models_dir, "best_numpy_dnn.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modelo {path} não encontrado.")
            model = robust_pickle_load(path)
            self.cached_models[model_type] = model
            return model
            
        elif model_type_lower in ["pytorch_dnn", "pt_dnn"]:
            path = os.path.join(self.models_dir, "best_pytorch_dnn.pt")
            input_dim = len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 1000
            model = TabularDNN(input_dim=input_dim).to(self.device)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.cached_models[model_type] = model
            return model
            
        elif model_type_lower in ["pytorch_bilstm", "pt_bilstm", "bilstm"]:
            path = os.path.join(self.models_dir, "best_pytorch_bilstm.pt")
            vocab_size = self.tokenizer.vocab_size if self.tokenizer else 10000
            model = TextBiLSTM(vocab_size=vocab_size).to(self.device)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.cached_models[model_type] = model
            return model
            
        elif model_type_lower in ["pytorch_transformer", "pt_transformer", "transformer"]:
            path = os.path.join(self.models_dir, "best_pytorch_transformer.pt")
            vocab_size = self.tokenizer.vocab_size if self.tokenizer else 10000
            model = TextTransformer(vocab_size=vocab_size).to(self.device)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.cached_models[model_type] = model
            return model
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    def predict_probabilities(self, texts, model_type="pytorch_bilstm"):
        """Retorna as probabilidades contínuas (0.0 a 1.0) de o texto ser gerado por IA."""
        if isinstance(texts, str):
            texts = [texts]
            
        model = self.get_model(model_type)
        model_type_lower = model_type.lower().strip()
        
        if model_type_lower.startswith("numpy"):
            if self.vectorizer is None:
                raise ValueError("Vetorizador TF-IDF não carregado.")
            X_tfidf = self.vectorizer.transform(texts).toarray()
            # NumPy model forward propagation em modo de teste (training=False)
            probs = model.forward_propagation(X_tfidf, training=False).flatten()
            return np.clip(probs, 0.0, 1.0)
            
        elif model_type_lower in ["pytorch_dnn", "pt_dnn"]:
            if self.vectorizer is None:
                raise ValueError("Vetorizador TF-IDF não carregado.")
            X_tfidf = self.vectorizer.transform(texts).toarray()
            X_tensor = torch.tensor(X_tfidf, dtype=torch.float32).to(self.device)
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            return probs
            
        elif model_type_lower in ["pytorch_bilstm", "pt_bilstm", "bilstm", "pytorch_transformer", "pt_transformer", "transformer"]:
            if self.tokenizer is None:
                raise ValueError("Tokenizador não carregado.")
            encoded = [self.tokenizer.encode(t) for t in texts]
            X_tensor = torch.tensor(encoded, dtype=torch.long).to(self.device)
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            return probs
            
        else:
            raise ValueError(f"Modelo não suportado: {model_type}")

    def predict_labels(self, texts, model_type="pytorch_bilstm", threshold=0.5):
        """Retorna labels binárias (0 = Human, 1 = AI) e probabilidade."""
        probs = self.predict_probabilities(texts, model_type=model_type)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    def predict_single_text(self, text, model_type="pytorch_bilstm"):
        """Inferência rápida para um único texto com metadados."""
        words = len(str(text).split())
        probs = self.predict_probabilities([text], model_type=model_type)
        prob_ai = float(probs[0])
        label_num = 1 if prob_ai >= 0.5 else 0
        label_name = "AI" if label_num == 1 else "Human"
        
        return {
            "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
            "word_count": words,
            "predicted_label": label_name,
            "label_num": label_num,
            "probability_ai": prob_ai,
            "probability_human": 1.0 - prob_ai,
            "model_used": model_type
        }

    def generate_submission(self, input_csv_path, output_csv_path, model_type="pytorch_bilstm", threshold=0.5):
        """
        Processa um ficheiro de teste fornecido pelo docente e gera as previsões.
        Aceita ficheiros com colunas 'ID' e 'text' (ou primeiras 2 colunas).
        """
        print(f"[*] A carregar ficheiro de teste: {input_csv_path}")
        df = pd.read_csv(input_csv_path, sep=None, engine='python', on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        
        # Identificar coluna ID
        id_col = None
        for col in df.columns:
            if col.lower() in ['id', 'index', 'guid']:
                id_col = col
                break
        if id_col is None:
            id_col = df.columns[0]
            
        # Identificar coluna Text
        text_col = None
        for col in df.columns:
            if col.lower() in ['text', 'content', 'body', 'input']:
                text_col = col
                break
        if text_col is None:
            text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
        texts = df[text_col].astype(str).tolist()
        ids = df[id_col].tolist()
        
        print(f"[*] A executar inferência com {model_type} em {len(texts)} exemplos...")
        preds, probs = self.predict_labels(texts, model_type=model_type, threshold=threshold)
        
        df_submission = pd.DataFrame({
            'ID': ids,
            'prediction': preds,
            'probability_ai': np.round(probs, 4),
            'label': ["ai" if p == 1 else "human" for p in preds]
        })
        
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df_submission.to_csv(output_csv_path, index=False)
        print(f"[+] Ficheiro de submissão gerado com sucesso: {output_csv_path}")
        print(f"    Distribuição prevista: Human = {np.sum(preds == 0)}, AI = {np.sum(preds == 1)}")
        return df_submission


if __name__ == "__main__":
    engine = ModelInferenceEngine()
    test_phrase = "In this study, we evaluate deep learning algorithms for text classification in biology and medicine."
    res = engine.predict_single_text(test_phrase, model_type="pytorch_bilstm")
    print("\nTeste rápido de inferência:")
    print(res)
