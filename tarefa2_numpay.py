# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:19:03 2026
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Adicionar a pasta 'src' ao caminho de busca do Python para carregar as classes do professor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Importar as classes criadas pelo professor nas aulas!
from data import Data
from layers import DenseLayer
from activation import SigmoidActivation, ReLUActivation
from losses import BinaryCrossEntropy
from optimizer import Optimizer
from neuralnetwork import NeuralNetwork
from metrics import accuracy

# -------------------------------------------------------------
# 1. CARREGAR OS DADOS LIMPOS (Gerados na Tarefa 1)
# -------------------------------------------------------------
path_train = os.path.join("data", "train_clean.csv")
path_val = os.path.join("data", "val_clean.csv")
path_test = os.path.join("data", "test_clean.csv")

train_df = pd.read_csv(path_train)
val_df = pd.read_csv(path_val)
test_df = pd.read_csv(path_test)

# Selecionamos um subconjunto para treinar de forma rápida e eficiente
X_train_raw = train_df['text'].astype(str).values[:15000]
y_train_raw = train_df['label_num'].values[:15000].reshape(-1, 1)

X_val_raw = val_df['text'].astype(str).values[:3000]
y_val_raw = val_df['label_num'].values[:3000].reshape(-1, 1)

X_test_raw = test_df['text'].astype(str).values[:3000]
y_test_raw = test_df['label_num'].values[:3000].reshape(-1, 1)

# -------------------------------------------------------------
# 2. TRANSFORMADOR DE TEXTO PARA MATRIZ NUMÉRICA (TF-IDF)
# -------------------------------------------------------------
print("Convertendo texto para matriz numérica com TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

X_train_num = vectorizer.fit_transform(X_train_raw).toarray()
X_val_num = vectorizer.transform(X_val_raw).toarray()
X_test_num = vectorizer.transform(X_test_raw).toarray()

# -------------------------------------------------------------
# 3. ENCAPSULAR NAS CLASSES 'Data' (data.py)
# -------------------------------------------------------------
train_dataset = Data(X=X_train_num, y=y_train_raw)
val_dataset = Data(X=X_val_num, y=y_val_raw)
test_dataset = Data(X=X_test_num, y=y_test_raw)

n_features = train_dataset.X.shape[1]
print(f"Dimensão das features (Número de termos TF-IDF): {n_features}")

# VERIFICAÇÃO DE TAMANHO
print(f"Total de exemplos no treino: {len(X_train_raw)}")
print(f"Total de exemplos na validação: {len(X_val_raw)}")
print(f"Total de exemplos no teste: {len(X_test_raw)}")

# VERIFICAÇÃO DE CLASSES
print(f"Número de exemplos Humanos (label 0) no treino: {np.sum(y_train_raw == 0)}")
print(f"Número de exemplos IA (label 1) no treino: {np.sum(y_train_raw == 1)}")

# -------------------------------------------------------------
# Função Auxiliar de Treino com Early Stopping
# -------------------------------------------------------------
def treinar_modelo(model, train_dataset, val_dataset, loss_fn, epochs=50, batch_size=64, name="Modelo"):
    print(f"\n--- A treinar {name} ---")
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Baralhar dados
        indices = np.arange(len(train_dataset.X))
        np.random.shuffle(indices)
        X_shuffled = train_dataset.X[indices]
        y_shuffled = train_dataset.y[indices]
        
        # Mini-batch Gradient Descent
        for i in range(0, len(X_shuffled), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward Pass
            y_pred = model.forward_propagation(X_batch, training=True)
            
            # Backward Pass
            loss_grad = loss_fn.derivative(y_batch, y_pred)
            for layer in reversed(model.layers):
                loss_grad = layer.backward_propagation(loss_grad)
                
        # Validação periódica a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            val_preds = model.predict(val_dataset)
            acc_val = accuracy(val_dataset.y, val_preds)
            print(f"Época {epoch+1}/{epochs} | Exatidão na Validação: {acc_val*100:.2f}%")
            
            # Lógica de Paragem Precoce
            if acc_val > best_acc:
                best_acc = acc_val
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 3:
                print("Paragem precoce ativada! O modelo parou de melhorar.")
                break
    return model

# -------------------------------------------------------------
# 4. CONSTRUIR E TREINAR A REGRESSÃO LOGÍSTICA (Baseline)
# -------------------------------------------------------------
loss_fn = BinaryCrossEntropy()

# Otimizador específico para a Regressão Logística
opt_lr = Optimizer(learning_rate=0.05, momentum=0.9)
net_lr = NeuralNetwork()
net_lr.loss_function = loss_fn
net_lr.optimizer = opt_lr

# Um neurónio na camada de saída com ativação Sigmoid
net_lr.add(DenseLayer(n_units=1, input_shape=(n_features,)))
net_lr.add(SigmoidActivation())

treinar_modelo(net_lr, train_dataset, val_dataset, loss_fn, epochs=50, batch_size=64, name="Regressão Logística (Baseline)")

# -------------------------------------------------------------
# 5. CONSTRUIR E TREINAR A REDE NEURONAL PROFUNDA (DNN)
# -------------------------------------------------------------
# Otimizador específico para a DNN
opt_dnn = Optimizer(learning_rate=0.05, momentum=0.9)
net_dnn = NeuralNetwork()
net_dnn.loss_function = loss_fn
net_dnn.optimizer = opt_dnn

# Adicionar camadas ocultas e de ativação
net_dnn.add(DenseLayer(n_units=64, input_shape=(n_features,)))
net_dnn.add(ReLUActivation())
net_dnn.add(DenseLayer(n_units=1))
net_dnn.add(SigmoidActivation())

treinar_modelo(net_dnn, train_dataset, val_dataset, loss_fn, epochs=50, batch_size=64, name="Rede Neuronal Profunda (DNN)")

# -------------------------------------------------------------
# 6. AVALIAÇÃO DOS MODELOS NO DATASET DE TESTE
# -------------------------------------------------------------
print("\n=== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ===")

test_preds_lr = net_lr.predict(test_dataset)
acc_test_lr = accuracy(test_dataset.y, test_preds_lr)
print(f"Exatidão da Regressão Logística (Baseline) no Teste: {acc_test_lr*100:.2f}%")

test_preds_dnn = net_dnn.predict(test_dataset)
acc_test_dnn = accuracy(test_dataset.y, test_preds_dnn)
print(f"Exatidão da Rede Neuronal (DNN) no Teste: {acc_test_dnn*100:.2f}%")

# -------------------------------------------------------------
# 7. GUARDAR OS MODELOS E PRÉ-PROCESSADOR NA PASTA 'models/'
# -------------------------------------------------------------
import pickle
os.makedirs("models", exist_ok=True)

with open(os.path.join("models", "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join("models", "best_numpy_lr.pkl"), "wb") as f:
    pickle.dump(net_lr, f)

with open(os.path.join("models", "best_numpy_dnn.pkl"), "wb") as f:
    pickle.dump(net_dnn, f)

print("\n[+] Modelos NumPy e Vetorizador TF-IDF guardados em 'models/' com sucesso!")