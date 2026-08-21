import os
import pandas as pd
from sklearn.model_selection import train_test_split

def carregar_e_preparar_dados():
    print("--- 1. Carregando os dados da pasta /data ---")
    
    path_inputs1 = os.path.join("data", "dataset1_inputs.csv")
    path_outputs1 = os.path.join("data", "dataset1_outputs.csv")
    path_dataset2 = os.path.join("data", "dataset2_complete.csv")
    
    # 1. Carregar Dataset 1
    if os.path.exists(path_inputs1) and os.path.exists(path_outputs1):
        # sep=None com engine='python' deteta automaticamente se é tabulação (\t) ou vírgula
        df_inputs1 = pd.read_csv(path_inputs1, sep=None, engine='python', on_bad_lines='skip')
        df_outputs1 = pd.read_csv(path_outputs1, sep=None, engine='python', on_bad_lines='skip')
        
        # Limpar espaços nos nomes das colunas
        df_inputs1.columns = [c.strip() for c in df_inputs1.columns]
        df_outputs1.columns = [c.strip() for c in df_outputs1.columns]
        
        df1 = pd.merge(df_inputs1, df_outputs1, on="ID")
        print(f"Dataset 1 carregado com sucesso! Total: {len(df1)}")
    else:
        print("Aviso: Ficheiros do Dataset 1 não encontrados em /data")
        df1 = pd.DataFrame()

    # 2. Carregar Dataset 2
    if os.path.exists(path_dataset2):
        df2 = pd.read_csv(path_dataset2, sep=None, engine='python', on_bad_lines='skip')
        df2.columns = [c.strip() for c in df2.columns]
        print(f"Dataset 2 carregado com sucesso! Total: {len(df2)}")
    else:
        print("Aviso: Ficheiro dataset2_complete.csv não encontrado em /data")
        df2 = pd.DataFrame()

    # 3. Mapear e Padronizar Nomes das Colunas
    for df in [df1, df2]:
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]

    # Concatenar
    df_total = pd.concat([df1, df2], ignore_index=True)
    
    # Garantir nomes padrão de colunas
    col_texto = 'text' if 'text' in df_total.columns else df_total.columns[1]
    col_label = 'label' if 'label' in df_total.columns else df_total.columns[-1]
    
    df_total = df_total.rename(columns={col_texto: 'text', col_label: 'label'})
    df_total = df_total.dropna(subset=['text', 'label']).drop_duplicates(subset=['text'])

    # Converter rótulos em binário (0 = human, 1 = ai)
    df_total['label_clean'] = df_total['label'].astype(str).str.lower()
    df_total['label_num'] = df_total['label_clean'].apply(lambda x: 1 if 'ai' in x or '1' in x else 0)

    print(f"\nTotal de dados válidos combinados: {len(df_total)}")

    # 4. Divisão (Train, Val, Test)
    train_df, temp_df = train_test_split(df_total, test_size=0.20, random_state=42, stratify=df_total['label_num'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label_num'])

    print("\n--- Divisão Concluída ---")
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")

    # Guardar dados processados
    train_df.to_csv(os.path.join("data", "train_clean.csv"), index=False)
    val_df.to_csv(os.path.join("data", "val_clean.csv"), index=False)
    test_df.to_csv(os.path.join("data", "test_clean.csv"), index=False)
    print("\nFicheiros limpos guardados com sucesso na pasta 'data/'!")

if __name__ == "__main__":
    carregar_e_preparar_dados()