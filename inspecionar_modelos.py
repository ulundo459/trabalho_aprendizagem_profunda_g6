# -*- coding: utf-8 -*-
"""
Script de Inspeção e Visualização de Modelos e Pré-processadores Guardados
UC: Aprendizagem Profunda - Trabalho Prático

Permite inspecionar o conteúdo detalhado de:
1. models/simple_tokenizer.pkl
2. models/tfidf_vectorizer.pkl
3. models/best_numpy_lr.pkl
4. models/best_numpy_dnn.pkl
5. models/best_pytorch_dnn.pt
6. models/best_pytorch_bilstm.pt
7. models/best_pytorch_transformer.pt
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Adicionar a pasta 'src' ao path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from inference import robust_pickle_load

MODELS_DIR = "models"


def separador(titulo):
    print("\n" + "=" * 75)
    print(f"  {titulo.upper()}")
    print("=" * 75)


# =====================================================================
# 1. INSPEÇÃO DO TOKENIZADOR (SimpleTokenizer)
# =====================================================================
def inspecionar_tokenizer():
    separador("1. Inspeção: models/simple_tokenizer.pkl")
    path = os.path.join(MODELS_DIR, "simple_tokenizer.pkl")
    
    if not os.path.exists(path):
        print(f"[!] Ficheiro {path} não encontrado.")
        return
        
    tok = robust_pickle_load(path)
    print(f"[*] Tipo de Objeto             : {type(tok).__name__}")
    print(f"[*] Tamanho Real do Vocabulário : {tok.vocab_size} tokens únicos")
    print(f"[*] Capacidade Máxima          : {tok.max_vocab_size} tokens")
    print(f"[*] Comprimento Fixo (Padding) : {tok.max_seq_len} tokens")
    
    print("\n--- Primeiros 20 Tokens Mapeados ---")
    df_vocab = pd.DataFrame(list(tok.vocab.items())[:20], columns=['Palavra / Token', 'ID Numérico'])
    print(df_vocab.to_string(index=False))
    
    # Teste de codificação
    frase_teste = "The neural network analyzes deep learning models for classification."
    tokens_enc = tok.encode(frase_teste)
    print(f"\n[*] Teste de Codificação:")
    print(f"  Texto de Entrada: \"{frase_teste}\"")
    print(f"  IDs Codificados : {tokens_enc[:12]}... (Total: {len(tokens_enc)} posições)")


# =====================================================================
# 2. INSPEÇÃO DO VETORIZADOR TF-IDF (TfidfVectorizer)
# =====================================================================
def inspecionar_tfidf():
    separador("2. Inspeção: models/tfidf_vectorizer.pkl")
    path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    
    if not os.path.exists(path):
        print(f"[!] Ficheiro {path} não encontrado.")
        return
        
    vec = robust_pickle_load(path)
    feature_names = vec.get_feature_names_out()
    
    print(f"[*] Tipo de Objeto             : {type(vec).__name__}")
    print(f"[*] Número de Features (Termos): {len(feature_names)}")
    print(f"[*] Stop Words Utilizadas      : {vec.stop_words if vec.stop_words else 'Nenhuma'}")
    print(f"[*] Intervalo de n-gramas      : {vec.ngram_range}")
    
    idf_pairs = sorted(zip(feature_names, vec.idf_), key=lambda x: x[1], reverse=True)
    df_idf_top = pd.DataFrame(idf_pairs[:10], columns=['Termo', 'Peso IDF (Maior)'])
    df_idf_low = pd.DataFrame(idf_pairs[-10:], columns=['Termo', 'Peso IDF (Menor / Frequente)'])
    
    print("\n--- 10 Termos com Maior Peso IDF (Mais Específicos) ---")
    print(df_idf_top.to_string(index=False))
    
    print("\n--- 10 Termos Mais Frequentes (Menor Peso IDF) ---")
    print(df_idf_low.to_string(index=False))


# =====================================================================
# 3. INSPEÇÃO DA REGRESSÃO LOGÍSTICA NUMPY (best_numpy_lr.pkl)
# =====================================================================
def inspecionar_numpy_lr():
    separador("3. Inspeção: models/best_numpy_lr.pkl (Regressão Logística)")
    path = os.path.join(MODELS_DIR, "best_numpy_lr.pkl")
    
    if not os.path.exists(path):
        print(f"[!] Ficheiro {path} não encontrado.")
        return
        
    lr = robust_pickle_load(path)
    print(f"[*] Tipo de Objeto             : {type(lr).__name__}")
    print(f"[*] Número de Camadas          : {len(lr.layers)}")
    
    dense = lr.layers[0]
    weights = dense.weights
    bias = dense.biases
    
    print(f"[*] Matriz de Pesos (W)        : Shape {weights.shape} ({weights.size} parâmetros)")
    print(f"[*] Termo Independente (Bias b): {bias.flatten().tolist()}")
    print(f"[*] Estatísticas dos Pesos     : Média = {np.mean(weights):.4f} | Desvio Padrão = {np.std(weights):.4f}")
    
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        vec = robust_pickle_load(vec_path)
        feature_names = vec.get_feature_names_out()
        w_flat = weights.flatten()
        
        top_ai = sorted(zip(feature_names, w_flat), key=lambda x: x[1], reverse=True)[:5]
        top_human = sorted(zip(feature_names, w_flat), key=lambda x: x[1])[:5]
        
        print("\n--- Palavras mais associadas a 'IA' pelos pesos deste modelo ---")
        for word, val in top_ai:
            print(f"  (+) '{word:<20}': peso = {val:+.4f}")
            
        print("\n--- Palavras mais associadas a 'Humano' pelos pesos deste modelo ---")
        for word, val in top_human:
            print(f"  (-) '{word:<20}': peso = {val:+.4f}")


# =====================================================================
# 4. INSPEÇÃO DA REDE PROFUNDA NUMPY (best_numpy_dnn.pkl)
# =====================================================================
def inspecionar_numpy_dnn():
    separador("4. Inspeção: models/best_numpy_dnn.pkl (Rede Neuronal Profunda)")
    path = os.path.join(MODELS_DIR, "best_numpy_dnn.pkl")
    
    if not os.path.exists(path):
        print(f"[!] Ficheiro {path} não encontrado.")
        return
        
    dnn = robust_pickle_load(path)
    print(f"[*] Tipo de Objeto             : {type(dnn).__name__}")
    print(f"[*] Total de Camadas na Rede   : {len(dnn.layers)}")
    
    total_params = 0
    print("\n--- Estrutura Camada a Camada ---")
    for i, layer in enumerate(dnn.layers):
        nome_camada = type(layer).__name__
        if hasattr(layer, 'weights') and layer.weights is not None:
            w_shape = layer.weights.shape
            b_shape = layer.biases.shape if hasattr(layer, 'biases') and layer.biases is not None else 'N/A'
            params = layer.weights.size + (layer.biases.size if hasattr(layer, 'biases') and layer.biases is not None else 0)
            total_params += params
            print(f"  Camada {i+1:02d}: {nome_camada:<22} | W: {str(w_shape):<12} | b: {str(b_shape):<10} | Params: {params}")
        else:
            print(f"  Camada {i+1:02d}: {nome_camada:<22} | (Função de Ativação)")
            
    print(f"\n[*] Total de Parâmetros Treináveis na DNN: {total_params:,}")


# =====================================================================
# 5. INSPEÇÃO DOS MODELOS PYTORCH (.PT)
# =====================================================================
def inspecionar_pytorch_models():
    separador("5. Inspeção dos Modelos PyTorch (.pt)")
    
    pt_files = [
        ("best_pytorch_dnn.pt", "PyTorch Tabular DNN (TF-IDF)"),
        ("best_pytorch_bilstm.pt", "PyTorch BiLSTM (Embeddings)"),
        ("best_pytorch_transformer.pt", "PyTorch Transformer Encoder")
    ]
    
    for filename, desc in pt_files:
        path = os.path.join(MODELS_DIR, filename)
        print(f"\n>>> MODELO: {desc} ({filename})")
        
        if not os.path.exists(path):
            print(f"    [!] Ficheiro {path} não encontrado.")
            continue
            
        file_size_kb = os.path.getsize(path) / 1024
        # Carregar o dicionário de pesos (state_dict) do PyTorch
        state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        
        total_params = 0
        layer_rows = []
        
        for key, tensor in state_dict.items():
            num_params = tensor.numel()
            total_params += num_params
            layer_rows.append({
                "Tensor / Camada": key,
                "Dimensões (Shape)": str(list(tensor.shape)),
                "Parâmetros": num_params,
                "Tipo de Dado": str(tensor.dtype).replace("torch.", "")
            })
            
        df_layers = pd.DataFrame(layer_rows)
        print(f"    • Tamanho em Disco     : {file_size_kb:.2f} KB ({file_size_kb/1024:.2f} MB)")
        print(f"    • Total de Parâmetros : {total_params:,} parâmetros")
        print(f"    • Total de Tensores   : {len(state_dict)}")
        print("\n--- Tensores e Camadas Internas ---")
        print(df_layers.to_string(index=False))
        print("-" * 70)


# =====================================================================
# 6. EXECUÇÃO PRINCIPAL
# =====================================================================
def main():
    print("#" * 75)
    print("      INSPECTOR DE MODELOS E ARTEFACTOS (.PKL E .PT)")
    print("#" * 75)
    
    inspecionar_tokenizer()
    inspecionar_tfidf()
    inspecionar_numpy_lr()
    inspecionar_numpy_dnn()
    inspecionar_pytorch_models()
    
    separador("Concluído com Sucesso")
    print("[+] Todos os ficheiros (.pkl e .pt) foram lidos e inspecionados com sucesso!\n")


if __name__ == "__main__":
    main()
