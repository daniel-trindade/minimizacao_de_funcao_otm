# visualizar um dataframe pandas
import pandas as pd
from IPython.display import display # Útil para notebooks Jupyter ou ambientes interativos
import os # Importar para usar os.path.exists

def importar_dataframe(nome_arquivo):
    """
    Função para importar um DataFrame do pandas a partir de um arquivo CSV.
    Tenta ler com UTF-8 e, em caso de erro de codificação, tenta com latin-1.
    
    Parâmetros:
    nome_arquivo (str): O caminho do arquivo CSV a ser importado.
    
    Retorna:
    pd.DataFrame: O DataFrame importado.
    """
    if not os.path.exists(nome_arquivo):
        print(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado. Verifique o caminho e o nome do arquivo.")
        return None

    # Etapa 1: Tentar determinar quantas linhas de comentário existem,
    # usando um fallback de codificação.
    skip_rows_count = 0
    encodings_to_try = ['utf-8', 'latin-1'] # Ordem de tentativa

    for encoding_attempt in encodings_to_try:
        try:
            with open(nome_arquivo, 'r', encoding=encoding_attempt) as f:
                temp_skip_count = 0
                for line in f:
                    if line.strip().startswith('#'):
                        temp_skip_count += 1
                    else:
                        break
                skip_rows_count = temp_skip_count # Se chegou aqui, esta codificação funcionou
                break # Sai do loop de tentativas de codificação para os comentários
        except UnicodeDecodeError:
            print(f"Aviso: Falha ao ler cabeçalho com '{encoding_attempt}'. Tentando a próxima codificação...")
            continue # Tenta a próxima codificação
        except Exception as e:
            print(f"Erro inesperado ao ler cabeçalho do arquivo com '{encoding_attempt}': {e}")
            return None
    else: # Este 'else' é executado se o loop 'for' não for interrompido por um 'break'
        print("Erro: Nenhuma das codificações tentadas conseguiu ler o cabeçalho do arquivo.")
        return None


    # Etapa 2: Tentar ler o CSV completo com as codificações
    for encoding_attempt in encodings_to_try:
        try:
            # Use o skip_rows_count determinado e a codificação atual
            df = pd.read_csv(nome_arquivo, skiprows=skip_rows_count, skipinitialspace=True, encoding=encoding_attempt)
            print(f"Arquivo '{nome_arquivo}' importado com sucesso usando codificação '{encoding_attempt}'.")
            return df
        except UnicodeDecodeError:
            print(f"Erro de codificação com '{encoding_attempt}' para '{nome_arquivo}'. Tentando a próxima codificação...")
            continue # Tenta a próxima codificação
        except pd.errors.EmptyDataError:
            print(f"Erro: O arquivo '{nome_arquivo}' está vazio ou contém apenas comentários após pular.")
            return None
        except Exception as e:
            print(f"Erro ao importar o arquivo com '{encoding_attempt}': {e}")
            return None
    
    print("Erro: Nenhuma das codificações tentadas conseguiu importar o DataFrame completo.")
    return None


def visualizar_dataframe(df):
    """
    Função para visualizar um DataFrame do pandas de forma mais amigável.
    
    Parâmetros:
    df (pd.DataFrame): O DataFrame a ser visualizado.
    """
    if isinstance(df, pd.DataFrame) and not df.empty:
        display(df) 
        # Alternativa para console, caso não esteja em Jupyter: print(df.to_string())
    else:
        print("O objeto fornecido não é um DataFrame do pandas ou está vazio.")

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Caminho para o seu arquivo CSV.
    # AJUSTE O NOME DO ARQUIVO AQUI para corresponder EXATAMENTE ao que foi gerado pelo analise_pso.py
    # (data, hora e número de runs).
    # Por exemplo:
    # nome_do_arquivo_csv = r"resultados_analise/Sumario_Desempenho_PSO_media_20_runs_20250702_204718.csv"
    # Certifique-se de usar barras normais '/' ou r"..." para o caminho.
    
    # Exemplo (você deve substituir este valor):
    # nome_do_arquivo_csv = r"resultados_analise\Sumario_Desempenho_PSO_media_3_runs_20250702_210655.csv" 
    nome_do_arquivo_csv = r"./resultados_analise_ag/sumario_estatistico/Sumario_Desempenho_AG_20251125_174433.csv"
    
    print(f"Tentando importar o arquivo: {nome_do_arquivo_csv}")
    
    # Importar o DataFrame
    meu_dataframe = importar_dataframe(nome_do_arquivo_csv)

    # Visualizar o DataFrame
    if meu_dataframe is not None:
        visualizar_dataframe(meu_dataframe)
    else:
        print("Não foi possível carregar o DataFrame para visualização.")

#%%
# %%
import pandas as pd
import os

def importar_dataframe(caminho_arquivo):
    """
    Importa um DataFrame de um arquivo CSV.
    Retorna o DataFrame ou None em caso de erro.
    """
    if not os.path.exists(caminho_arquivo):
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"DataFrame importado com sucesso de '{caminho_arquivo}'.")
        return df
    except Exception as e:
        print(f"Erro ao importar o DataFrame de '{caminho_arquivo}': {e}")
        return None

def visualizar_dataframe(df, num_linhas=5):
    """
    Visualiza as primeiras e últimas linhas de um DataFrame,
    e exibe informações gerais sobre ele.
    """
    if df is None:
        print("DataFrame vazio ou não carregado.")
        return

    print("\n--- Primeiras linhas do DataFrame ---")
    print(df.head(num_linhas))

    print("\n--- Últimas linhas do DataFrame ---")
    print(df.tail(num_linhas))

    print("\n--- Informações do DataFrame ---")
    df.info()

    print("\n--- Estatísticas Descritivas do DataFrame ---")
    print(df.describe())
