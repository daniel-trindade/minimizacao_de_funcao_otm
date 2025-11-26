# analise_ag.py

import numpy as np # Importa a biblioteca NumPy, essencial para operações numéricas, especialmente com arrays.
import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib para a criação de gráficos e visualizações.
import os # Importa o módulo 'os' para interagir com o sistema operacional, como criar diretórios.
from datetime import datetime # Importa a classe 'datetime' para trabalhar com datas e horas, útil para criar timestamps únicos.
import pandas as pd # Importa a biblioteca Pandas, usada para manipulação e análise de dados tabulares (DataFrames).

# Importe a função principal do seu algoritmo genético.
from Genetico import algoritmo_genetico # Importa a função `algoritmo_genetico` do arquivo `Genetico.py`, que deve conter a implementação do AG.

# --- Definição de Parâmetros para Análise do AG ---
# Dicionário que armazena os parâmetros de configuração para o algoritmo genético.
PARAMETROS_AG = { 
    "tamanho_populacao": 35, # Número de indivíduos na população de cada geração.
    "limites": (-500, 500), # Intervalo de valores para as variáveis de decisão (e.g., -500 a 500 para x e y).
    "num_geracoes": 200, # Número máximo de gerações que o AG irá rodar.
    "taxa_cruzamento": 0.7, # Probabilidade de ocorrer o cruzamento (crossover) entre indivíduos.
    "taxa_mutacao": 0.07, # Probabilidade de ocorrer mutação em um gene de um indivíduo.
    "geracoes_sem_melhora_limite": 20, # Critério de parada: número de gerações sem melhora significativa do melhor indivíduo global.
    "tolerancia": 1e-6 # Tolerância para o critério de parada (quão pequena a mudança no melhor valor deve ser para ser considerada "sem melhora").
}

NUM_SIMULACOES = 20 # Define quantas vezes o algoritmo genético será executado para a análise estatística.
OUTPUT_FOLDER_ANALYSIS = "resultados_analise_ag" # Define o nome da pasta principal onde todos os resultados da análise serão salvos.
OUTPUT_FOLDER_SUMMARY = os.path.join(OUTPUT_FOLDER_ANALYSIS, "sumario_estatistico") # Define o caminho para uma subpasta específica para salvar o CSV sumarizado.

class AnaliseAG: # Define uma classe para encapsular a lógica da análise do Algoritmo Genético.
    def __init__(self, parametros_ag, num_simulacoes, output_folder): # Método construtor da classe.
        self.parametros_ag = parametros_ag # Armazena os parâmetros do AG.
        self.num_simulacoes = num_simulacoes # Armazena o número de simulações a serem executadas.
        self.output_folder = output_folder # Armazena o diretório de saída principal.
        self.output_folder_summary = OUTPUT_FOLDER_SUMMARY # Armazena o diretório de saída para o sumário.
        
        os.makedirs(self.output_folder, exist_ok=True) # Cria o diretório principal de saída se ele não existir (exist_ok=True evita erro se já existir).
        os.makedirs(self.output_folder_summary, exist_ok=True) # Cria o subdiretório para o sumário se ele não existir.

    def executar(self): # Método principal que orquestra a execução das simulações e a análise.
        print(f"Iniciando análise do Algoritmo Genético com {self.num_simulacoes} simulações...\n") # Mensagem inicial.

        resultados_simulacoes = [] # Lista para armazenar os resultados detalhados de cada simulação individual.
        
        # NOVAS LISTAS para armazenar os históricos de cada simulação (para o gráfico de convergência).
        all_historico_melhor_geracao = [] # Armazena o histórico do melhor indivíduo em cada geração.
        all_historico_melhor_global = [] # Armazena o histórico do melhor indivíduo encontrado globalmente até cada geração.

        for i in range(self.num_simulacoes): # Loop para executar o AG múltiplas vezes.
            print(f"Executando simulação AG {i+1}/{self.num_simulacoes}...") # Mensagem de progresso.
            
            # Chama a função do algoritmo genético com os parâmetros definidos.
            results_ag = algoritmo_genetico( 
                tamanho_populacao=self.parametros_ag["tamanho_populacao"],
                limites=self.parametros_ag["limites"],
                num_geracoes=self.parametros_ag["num_geracoes"],
                taxa_cruzamento=self.parametros_ag["taxa_cruzamento"],
                taxa_mutacao=self.parametros_ag["taxa_mutacao"],
                geracoes_sem_melhora_limite=self.parametros_ag["geracoes_sem_melhora_limite"],
                tolerancia=self.parametros_ag["tolerancia"]
            )
            
            # Adiciona os resultados chave da simulação atual à lista de resultados_simulacoes.
            resultados_simulacoes.append({ 
                "Simulacao": i + 1, # Número da simulação.
                "Melhor Valor Encontrado": results_ag["melhor_valor_global"], # O melhor valor da função objetivo encontrado.
                "Gerações para Convergência": results_ag["iteracoes_executadas"], # Número de gerações até a convergência.
                "Avaliações Função Objetivo (Total)": results_ag["avaliacoes_funcao_total"], # Contagem total de avaliações da função objetivo.
                "Multiplicações (Total)": results_ag["multiplicacoes_total"], # Contagem total de multiplicações.
                "Divisões (Total)": results_ag["divisoes_total"], # Contagem total de divisões.
                "Avaliações (no Melhor Global)": results_ag["avaliacoes_minimo_global"], # Contagem de avaliações até o melhor global ser encontrado.
                "Multiplicações (no Melhor Global)": results_ag["multiplicacoes_minimo_global"], # Contagem de multiplicações até o melhor global ser encontrado.
                "Divisões (no Melhor Global)": results_ag["divisoes_minimo_global"], # Contagem de divisões até o melhor global ser encontrado.
            })
            
            # Coleta os dois históricos de convergência de cada simulação para posterior plotagem de média.
            all_historico_melhor_geracao.append(results_ag["historico_melhor_geracao"]) 
            all_historico_melhor_global.append(results_ag["historico_melhor_global"]) 


        # --- Geração de Estatísticas e Relatórios (sem alterações nesta parte) ---
        # Converte a lista de resultados em um DataFrame do Pandas para facilitar a análise.
        df_resultados_completos = pd.DataFrame(resultados_simulacoes) 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp para nomes de arquivos.
        
        # Define os nomes dos arquivos CSV e Excel para os resultados brutos.
        csv_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.csv") 
        excel_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.xlsx") 
        
        df_resultados_completos.to_csv(csv_filename_bruto, index=False) # Salva o DataFrame completo em um arquivo CSV.
        
        try: # Tenta salvar em formato Excel.
            df_resultados_completos.to_excel(excel_filename_bruto, index=False) # Salva o DataFrame completo em um arquivo Excel.
            print(f"Resultados detalhados de cada simulação salvos em Excel: {excel_filename_bruto}") # Mensagem de sucesso.
        except ModuleNotFoundError: # Captura o erro se a biblioteca 'openpyxl' não estiver instalada.
            print("AVISO: openpyxl não está instalado. Não foi possível salvar os resultados em formato Excel.") # Aviso ao usuário.
            print("Para habilitar o salvamento em Excel, execute: pip install openpyxl") # Instrução para instalação.
        
        print(f"\nResultados detalhados de cada simulação salvos em CSV: {csv_filename_bruto}") # Mensagem de sucesso para CSV.


        # Seleciona as colunas de métricas para calcular as estatísticas descritivas (média e desvio padrão).
        colunas_metrica = [col for col in df_resultados_completos.columns if col != 'Simulacao'] 
        # Calcula as estatísticas descritivas ('mean' e 'std') para as colunas de métricas.
        stats = df_resultados_completos[colunas_metrica].describe().loc[['mean', 'std']].transpose() 
        stats.rename(columns={'mean': 'Media', 'std': 'Desvio_Padrao'}, inplace=True) # Renomeia as colunas para "Media" e "Desvio_Padrao".
        df_sumario_ag = stats.reset_index() # Reseta o índice e transforma o índice anterior em uma coluna.
        df_sumario_ag.rename(columns={'index': 'Métrica'}, inplace=True) # Renomeia a coluna do índice para "Métrica".

        # Define a ordem desejada para as métricas no sumário.
        ordem_metrica = [ 
            "Melhor Valor Encontrado",
            "Gerações para Convergência",
            "Avaliações Função Objetivo (Total)",
            "Multiplicações (Total)",
            "Divisões (Total)",
            "Avaliações (no Melhor Global)",
            "Multiplicações (no Melhor Global)",
            "Divisões (no Melhor Global)"
        ]
        # Cria uma lista `ordem_final` garantindo que apenas métricas existentes sejam incluídas e que a ordem seja mantida.
        ordem_final = [m for m in ordem_metrica if m in df_sumario_ag['Métrica'].values] 
        # Adiciona quaisquer outras métricas que não estavam na ordem_metrica predefinida.
        for m in df_sumario_ag['Métrica'].values: 
            if m not in ordem_final:
                ordem_final.append(m)
        
        # Converte a coluna 'Métrica' para um tipo categórico ordenado, aplicando a ordem definida.
        df_sumario_ag['Métrica'] = pd.Categorical(df_sumario_ag['Métrica'], categories=ordem_final, ordered=True) 
        df_sumario_ag = df_sumario_ag.sort_values('Métrica') # Ordena o DataFrame pela coluna 'Métrica'.
        df_sumario_ag.reset_index(drop=True, inplace=True) # Reseta o índice novamente.

        # Define o nome do arquivo CSV para o sumário de desempenho.
        sumario_csv_filename = os.path.join(self.output_folder_summary, f"Sumario_Desempenho_AG_{timestamp}.csv") 
        df_sumario_ag.to_csv(sumario_csv_filename, index=False) # Salva o DataFrame do sumário em CSV.
        print(f"\nSumário de desempenho AG (Média e Desvio Padrão) salvo em: {sumario_csv_filename}") # Mensagem de sucesso.
        print("\n--- Sumário de Desempenho AG ---") # Título para a exibição no console.
        print(df_sumario_ag) # Imprime o DataFrame do sumário no console.

        # Gera um relatório de texto com todas as informações da análise.
        report_filename = os.path.join(self.output_folder, f"relatorio_analise_ag_{timestamp}.txt") # Define o nome do arquivo do relatório.
        with open(report_filename, "w") as f: # Abre o arquivo em modo de escrita.
            f.write(f"Análise do Algoritmo Genético ({self.num_simulacoes} simulações)\n") # Escreve o título do relatório.
            f.write("------------------------------------------------------------------\n") # Separador.
            f.write("Parâmetros do AG:\n") # Seção de parâmetros.
            for param, value in self.parametros_ag.items(): # Itera sobre os parâmetros do AG.
                f.write(f"  {param.replace('_', ' ').capitalize()}: {value}\n") # Escreve cada parâmetro formatado.
            f.write("------------------------------------------------------------------\n") # Separador.
            f.write("Estatísticas Sumárias das Simulações (Formato 'describe'):\n") # Seção de estatísticas detalhadas.
            # Escreve um resumo estatístico completo (média, std, min, max) das métricas.
            f.write(df_resultados_completos.describe().loc[['mean', 'std', 'min', 'max']].transpose().to_string()) 
            f.write("\n------------------------------------------------------------------\n") # Separador.
            f.write("\nSumário de Desempenho (Média e Desvio Padrão):\n") # Seção do sumário.
            f.write(df_sumario_ag.to_string()) # Escreve o sumário de média e desvio padrão.
            f.write("\n------------------------------------------------------------------\n") # Separador final.
            print(f"Relatório de análise salvo em: {report_filename}") # Mensagem de sucesso.

        # Garantir que todos os históricos tenham o mesmo comprimento para calcular média/desvio
        # Encontra o comprimento máximo entre todos os históricos de melhor global.
        max_len = max(len(h) for h in all_historico_melhor_global) 
        if not all_historico_melhor_global: # Adiciona esta verificação para garantir que há dados para plotar.
            print("Nenhum histórico de dados para plotar o gráfico de convergência AG.") # Mensagem de aviso.
            return # Sai da função se não houver dados.

        padded_historico_melhor_geracao = [] # Lista para históricos de melhor geração preenchidos.
        padded_historico_melhor_global = [] # Lista para históricos de melhor global preenchidos.

        for i in range(self.num_simulacoes): # Itera sobre cada simulação para preencher os históricos.
            hg_geracao = all_historico_melhor_geracao[i] # Obtém o histórico de melhor geração da simulação atual.
            hg_global = all_historico_melhor_global[i] # Obtém o histórico de melhor global da simulação atual.

            # Preenche o histórico de melhor geração com o último valor para que todos tenham o mesmo comprimento.
            padded_historico_melhor_geracao.append( 
                hg_geracao + [hg_geracao[-1]] * (max_len - len(hg_geracao)) if hg_geracao else [np.nan] * max_len
            )
            # Preenche o histórico de melhor global com o último valor para que todos tenham o mesmo comprimento.
            padded_historico_melhor_global.append( 
                hg_global + [hg_global[-1]] * (max_len - len(hg_global)) if hg_global else [np.nan] * max_len
            )

        # Calcula a média e o desvio padrão dos históricos preenchidos para as plotagens.
        mean_historico_melhor_geracao = np.nanmean(padded_historico_melhor_geracao, axis=0) # Média do melhor valor por geração.
        std_historico_melhor_geracao = np.nanstd(padded_historico_melhor_geracao, axis=0) # Desvio padrão do melhor valor por geração.
        
        mean_historico_melhor_global = np.nanmean(padded_historico_melhor_global, axis=0) # Média do melhor valor global encontrado.

        geracoes = np.arange(1, max_len + 1) # Cria um array para o eixo X (número de gerações).
        
        # Estilo e plotagem idênticos ao PSO (reutiliza o estilo e a estrutura de plotagem).
        plt.style.use('seaborn-v0_8-whitegrid') # Define o estilo visual do gráfico.
        fig, ax = plt.subplots(figsize=(12, 7)) # Cria a figura e os eixos do gráfico.

        # Linha azul tracejada (Média do Melhor Z da Geração)
        # Plota a média dos melhores valores de aptidão encontrados em cada geração.
        ax.plot(geracoes, mean_historico_melhor_geracao, 'b--', label=f'Média do Melhor Z da Geração (AG)') 
        
        # Área sombreada do desvio padrão
        # Calcula os limites superior e inferior para a área de desvio padrão.
        upper_bound_geracao = mean_historico_melhor_geracao + std_historico_melhor_geracao 
        lower_bound_geracao = mean_historico_melhor_geracao - std_historico_melhor_geracao 
        # Preenche a área entre os limites para visualizar o desvio padrão.
        ax.fill_between(geracoes, lower_bound_geracao, upper_bound_geracao, 
                         color='blue', alpha=0.1, label='Desvio Padrão (±1σ)') 
        
        # Linha vermelha sólida (Média do Melhor Valor Global Encontrado)
        # Plota a média dos melhores valores de aptidão encontrados globalmente ao longo das gerações.
        ax.plot(geracoes, mean_historico_melhor_global, 'r-', label='Média do Melhor Valor Global Encontrado', linewidth=2) 

        # Define o título do gráfico, incluindo o nome do algoritmo e o número de execuções.
        ax.set_title(f'Gráfico de Convergência - Algoritmo Genético (Média de {self.num_simulacoes} Execuções)', fontsize=16) 
        ax.set_xlabel('Número de Iterações/Gerações', fontsize=12) # Define o rótulo do eixo X.
        ax.set_ylabel('Valor da Função Objetivo (Z Ótimo)', fontsize=12) # Define o rótulo do eixo Y.
        
        ax.legend(loc='upper right', fontsize=10) # Adiciona a legenda ao gráfico.
        ax.grid(True) # Adiciona a grade ao gráfico.
        ax.set_ylim(bottom=None) # Ajusta o limite inferior do eixo Y automaticamente.

        plt.tight_layout() # Ajusta o layout do gráfico para evitar sobreposições.
        
        # Define o nome do arquivo para salvar o gráfico de convergência.
        plot_filename = os.path.join(self.output_folder, f"convergencia_media_ag_{timestamp}.png") 
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # Salva o gráfico em um arquivo PNG com alta resolução.
        plt.close() # Fecha a figura para liberar memória (importante em loops).
        print(f"Gráfico de convergência média AG salvo em: {plot_filename}") # Mensagem de sucesso.

# Bloco principal para executar a análise
if __name__ == "__main__": # Garante que o código dentro deste bloco só será executado quando o script for chamado diretamente.
    analisador = AnaliseAG(PARAMETROS_AG, NUM_SIMULACOES, OUTPUT_FOLDER_ANALYSIS) # Instancia a classe AnaliseAG.
    analisador.executar() # Chama o método `executar` para iniciar o processo de análise.