# analise_ag.py

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

# Importe a função principal do seu algoritmo genético.
from Genetico import algoritmo_genetico

# --- Definição de Parâmetros para Análise do AG ---
PARAMETROS_AG = { 
    "tamanho_populacao": 50, 
    "limites": (-500, 500), 
    "num_geracoes": 200, 
    "taxa_cruzamento": 0.8, 
    "taxa_mutacao": 0.2, 
    "geracoes_sem_melhora_limite": 20, 
    "tolerancia": 1e-6 
}

NUM_SIMULACOES = 10
OUTPUT_FOLDER_ANALYSIS = "resultados_analise_ag"
OUTPUT_FOLDER_SUMMARY = os.path.join(OUTPUT_FOLDER_ANALYSIS, "sumario_estatistico")

# --- FUNÇÃO PARA PLOTAR OS GRÁFICOS DE BARRAS (ADICIONADA) ---
def plot_bar_graphs(results_df, algorithm_name="AG", num_runs=1, params_display=None):
    """
    Gera gráficos de barra para as estatísticas de avaliações e operações computacionais.
    Adaptado para as colunas específicas do AG.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Calcula médias e desvios padrão baseados nos nomes das colunas do AG
    mean_best_global = results_df['Melhor Valor Encontrado'].mean()
    std_best_global = results_df['Melhor Valor Encontrado'].std()
    mean_iter_executed = results_df['Gerações para Convergência'].mean()
    std_iter_executed = results_df['Gerações para Convergência'].std()

    mean_fo_evals_total = results_df['Avaliações Função Objetivo (Total)'].mean()
    std_fo_evals_total = results_df['Avaliações Função Objetivo (Total)'].std()
    mean_mult_total = results_df['Multiplicações (Total)'].mean()
    std_mult_total = results_df['Multiplicações (Total)'].std()
    mean_div_total = results_df['Divisões (Total)'].mean()
    std_div_total = results_df['Divisões (Total)'].std()

    mean_fo_evals_min_global = results_df['Avaliações (no Melhor Global)'].mean()
    std_fo_evals_min_global = results_df['Avaliações (no Melhor Global)'].std()
    mean_mult_min_global = results_df['Multiplicações (no Melhor Global)'].mean()
    std_mult_min_global = results_df['Multiplicações (no Melhor Global)'].std()
    mean_div_min_global = results_df['Divisões (no Melhor Global)'].mean()
    std_div_min_global = results_df['Divisões (no Melhor Global)'].std()

    # --- Criação dos Gráficos de Barra ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f'Análise de Desempenho - {algorithm_name} (Média de {num_runs} Execuções)', fontsize=18)

    # Gráfico 1: Melhor Valor Global e Iterações
    ax1 = axes[0, 0]
    labels1 = ['Melhor Valor Global', 'Gerações Executadas']
    values1 = [mean_best_global, mean_iter_executed]
    errors1 = [std_best_global, std_iter_executed]
    colors1 = ['skyblue', 'lightcoral']
    bars1 = ax1.bar(labels1, values1, yerr=errors1, capsize=5, color=colors1)
    ax1.set_title('Média e Desvio Padrão (Melhor Valor e Gerações)', fontsize=14)
    ax1.set_ylabel('Valor / Gerações', fontsize=12)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval), 
                 f'{yval:.2f}', ha='center', va='bottom' if yval > 0 else 'top')

    # Gráfico 2: Avaliações e Operações - Total
    ax2 = axes[0, 1]
    labels2 = ['Avaliações Função', 'Multiplicações', 'Divisões']
    values2 = [mean_fo_evals_total, mean_mult_total, mean_div_total]
    errors2 = [std_fo_evals_total, std_mult_total, std_div_total]
    colors2 = ['lightgreen', 'salmon', 'plum']
    bars2 = ax2.bar(labels2, values2, yerr=errors2, capsize=5, color=colors2)
    ax2.set_title('Média e Desvio Padrão (Custo Total)', fontsize=14)
    ax2.set_ylabel('Quantidade', fontsize=12)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval), 
                 f'{yval:.0f}', ha='center', va='bottom' if yval > 0 else 'top')

    # Gráfico 3: Avaliações e Operações - Até achar o Mínimo
    ax3 = axes[1, 0]
    labels3 = ['Avaliações', 'Multiplicações', 'Divisões']
    values3 = [mean_fo_evals_min_global, mean_mult_min_global, mean_div_min_global]
    errors3 = [std_fo_evals_min_global, std_mult_min_global, std_div_min_global]
    colors3 = ['orange', 'darkorange', 'mediumpurple']
    bars3 = ax3.bar(labels3, values3, yerr=errors3, capsize=5, color=colors3)
    ax3.set_title('Média e Desvio Padrão (Custo até o Mínimo Global)', fontsize=14, pad=20)
    ax3.set_ylabel('Quantidade', fontsize=12)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval), 
                 f'{yval:.0f}', ha='center', va='bottom' if yval > 0 else 'top')

    # Texto com Parâmetros
    ax4 = axes[1, 1]
    ax4.set_axis_off()
    if params_display:
        params_text = "Parâmetros da Simulação (AG):\n"
        for key, value in params_display.items():
            params_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        ax4.text(0.5, 0.9, params_text, transform=ax4.transAxes, 
                 fontsize=11, verticalalignment='top', horizontalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Tenta usar self.output_folder se estiver dentro da classe, mas como a função está fora,
    # vamos salvar na pasta passada ou criar um nome genérico
    output_folder = "resultados_analise_ag"
    os.makedirs(output_folder, exist_ok=True)
    
    image_name = f"Graficos_Barras_{algorithm_name}_media_{num_runs}_runs_{timestamp}.png"
    image_path = os.path.join(output_folder, image_name)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Gráficos de barra salvos em: {image_path}")
    plt.close() # Fecha para não ficar aberto na memória


class AnaliseAG:
    def __init__(self, parametros_ag, num_simulacoes, output_folder):
        self.parametros_ag = parametros_ag
        self.num_simulacoes = num_simulacoes
        self.output_folder = output_folder
        self.output_folder_summary = OUTPUT_FOLDER_SUMMARY
        
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_folder_summary, exist_ok=True)

    def executar(self):
        print(f"Iniciando análise do Algoritmo Genético com {self.num_simulacoes} simulações...\n")

        resultados_simulacoes = []
        all_historico_melhor_geracao = []
        all_historico_melhor_global = []

        for i in range(self.num_simulacoes):
            print(f"Executando simulação AG {i+1}/{self.num_simulacoes}...")
            
            # Chama a função do algoritmo genético.
            results_ag = algoritmo_genetico( 
                tamanho_populacao=self.parametros_ag["tamanho_populacao"],
                limites=self.parametros_ag["limites"],
                num_geracoes=self.parametros_ag["num_geracoes"],
                taxa_cruzamento=self.parametros_ag["taxa_cruzamento"],
                taxa_mutacao=self.parametros_ag["taxa_mutacao"],
                geracoes_sem_melhora_limite=self.parametros_ag["geracoes_sem_melhora_limite"],
                tolerancia=self.parametros_ag["tolerancia"]
            )
            
            resultados_simulacoes.append({ 
                "Simulacao": i + 1,
                "Melhor Valor Encontrado": results_ag["melhor_valor_global"],
                "Gerações para Convergência": results_ag["iteracoes_executadas"],
                "Avaliações Função Objetivo (Total)": results_ag["avaliacoes_funcao_total"],
                "Multiplicações (Total)": results_ag["multiplicacoes_total"],
                "Divisões (Total)": results_ag["divisoes_total"],
                "Avaliações (no Melhor Global)": results_ag["avaliacoes_minimo_global"],
                "Multiplicações (no Melhor Global)": results_ag["multiplicacoes_minimo_global"],
                "Divisões (no Melhor Global)": results_ag["divisoes_minimo_global"],
            })
            
            all_historico_melhor_geracao.append(results_ag["historico_melhor_geracao"]) 
            all_historico_melhor_global.append(results_ag["historico_melhor_global"]) 

        # --- GERAÇÃO DE RELATÓRIOS E ESTATÍSTICAS ---
        print("\nGerando relatórios e gráfico de convergência final...")
        
        df_resultados_completos = pd.DataFrame(resultados_simulacoes) 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.csv") 
        df_resultados_completos.to_csv(csv_filename_bruto, index=False)
        
        # Gera Sumário
        colunas_metrica = [col for col in df_resultados_completos.columns if col != 'Simulacao'] 
        stats = df_resultados_completos[colunas_metrica].describe().loc[['mean', 'std']].transpose() 
        stats.rename(columns={'mean': 'Media', 'std': 'Desvio_Padrao'}, inplace=True)
        df_sumario_ag = stats.reset_index()
        df_sumario_ag.rename(columns={'index': 'Métrica'}, inplace=True)

        sumario_csv_filename = os.path.join(self.output_folder_summary, f"Sumario_Desempenho_AG_{timestamp}.csv") 
        df_sumario_ag.to_csv(sumario_csv_filename, index=False)
        print(f"Sumário salvo em: {sumario_csv_filename}")

        # --- NOVO: PLOTAGEM DOS GRÁFICOS DE BARRA ---
        # Chamada da função que adicionamos acima
        plot_bar_graphs(df_resultados_completos, algorithm_name="AG", num_runs=self.num_simulacoes, params_display=self.parametros_ag)

        # --- PLOTAGEM DO GRÁFICO DE CONVERGÊNCIA MÉDIA ---
        if not all_historico_melhor_global:
            return

        max_len = max(len(h) for h in all_historico_melhor_global) 
        padded_historico_melhor_geracao = []
        padded_historico_melhor_global = []

        for i in range(self.num_simulacoes):
            hg_geracao = all_historico_melhor_geracao[i]
            hg_global = all_historico_melhor_global[i]

            padded_historico_melhor_geracao.append( 
                hg_geracao + [hg_geracao[-1]] * (max_len - len(hg_geracao)) if hg_geracao else [np.nan] * max_len
            )
            padded_historico_melhor_global.append( 
                hg_global + [hg_global[-1]] * (max_len - len(hg_global)) if hg_global else [np.nan] * max_len
            )

        mean_historico_melhor_geracao = np.nanmean(padded_historico_melhor_geracao, axis=0)
        std_historico_melhor_geracao = np.nanstd(padded_historico_melhor_geracao, axis=0)
        mean_historico_melhor_global = np.nanmean(padded_historico_melhor_global, axis=0)

        geracoes = np.arange(1, max_len + 1)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(geracoes, mean_historico_melhor_geracao, 'b--', label=f'Média do Melhor Z da Geração (AG)') 
        
        upper_bound_geracao = mean_historico_melhor_geracao + std_historico_melhor_geracao 
        lower_bound_geracao = mean_historico_melhor_geracao - std_historico_melhor_geracao 
        ax.fill_between(geracoes, lower_bound_geracao, upper_bound_geracao, color='blue', alpha=0.1, label='Desvio Padrão (±1σ)') 
        
        ax.plot(geracoes, mean_historico_melhor_global, 'r-', label='Média do Melhor Valor Global Encontrado', linewidth=2) 

        ax.set_title(f'Gráfico de Convergência - Algoritmo Genético (Média de {self.num_simulacoes} Execuções)', fontsize=16) 
        ax.set_xlabel('Número de Iterações/Gerações', fontsize=12)
        ax.set_ylabel('Valor da Função Objetivo (Z Ótimo)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10) 
        ax.grid(True)
        ax.set_ylim(bottom=None)

        plt.tight_layout()
        plot_filename = os.path.join(self.output_folder, f"convergencia_media_ag_{timestamp}.png") 
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico de convergência média AG salvo em: {plot_filename}")

if __name__ == "__main__":
    analisador = AnaliseAG(PARAMETROS_AG, NUM_SIMULACOES, OUTPUT_FOLDER_ANALYSIS)
    analisador.executar()