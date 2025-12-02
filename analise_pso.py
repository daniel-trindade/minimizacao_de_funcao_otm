# analise_pso.py

import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib, usada para criar visualizações gráficas (gráficos).
import numpy as np # Importa a biblioteca NumPy, essencial para operações numéricas eficientes, especialmente com arrays.
import os # Importa o módulo 'os', que fornece uma maneira de interagir com o sistema operacional, como criar diretórios e manipular caminhos de arquivo.
from datetime import datetime # Importa a classe 'datetime' do módulo 'datetime', usada para trabalhar com datas e horas, como gerar timestamps para nomes de arquivos.
import pandas as pd # Importa a biblioteca Pandas, amplamente utilizada para manipulação e análise de dados, especialmente com DataFrames (tabelas de dados).

# Importa a classe PSO e a função objetivo w29+w1
# Certifique-se de que PSO.py e funcoes_otimizacao.py estão no mesmo diretório
from PSO import PSO # Importa a classe 'PSO' do arquivo 'PSO.py', que contém a implementação do algoritmo Particle Swarm Optimization.
from funcoes_otimizacao import minha_funcao_w29_w1 # Importa a função 'minha_funcao_w29_w1' do arquivo 'funcoes_otimizacao.py', que é a função objetivo a ser otimizada (minimizada neste caso).

def plot_convergence_graph(all_historico_melhor_global, all_historico_media_melhores_locais,
                           all_historico_desvio_padrao_melhores_locais, algorithm_name="PSO", num_runs=1):
    """
    Gera o gráfico de convergência, calculando a média e desvio padrão
    dos históricos de múltiplas execuções.
    """
    # Verifica se a lista de históricos do melhor global não está vazia. Se estiver, significa que não há dados para plotar.
    if not all_historico_melhor_global: 
        print("Nenhum histórico de dados para plotar o gráfico de convergência.") # Imprime uma mensagem de aviso.
        return # Sai da função se não houver dados.

    # Determina o comprimento máximo entre todos os históricos de 'media_melhores_locais'. 
    # Isso é necessário porque diferentes execuções do PSO podem parar em iterações distintas devido a critérios de convergência.
    max_len = max(len(h) for h in all_historico_media_melhores_locais)

    # Inicializa listas vazias que armazenarão os históricos após o "preenchimento" (padding).
    padded_historico_melhor_global = [] 
    padded_historico_media_melhores_locais = [] 
    padded_historico_desvio_padrao_melhores_locais = [] 

    # Itera sobre cada uma das 'num_runs' execuções para processar seus históricos.
    for i in range(num_runs): 
        hg = all_historico_melhor_global[i] # Obtém o histórico do melhor global para a simulação atual.
        hm = all_historico_media_melhores_locais[i] # Obtém o histórico da média dos melhores locais para a simulação atual.
        hdp = all_historico_desvio_padrao_melhores_locais[i] # Obtém o histórico do desvio padrão dos melhores locais para a simulação atual.

        # Adiciona o histórico do melhor global preenchido. Se o histórico for mais curto que `max_len`, ele é estendido
        # com o último valor válido. Se estiver vazio, é preenchido com NaNs (Not a Number).
        padded_historico_melhor_global.append( 
            hg + [hg[-1]] * (max_len - len(hg)) if hg else [np.nan] * max_len 
        )
        # Adiciona o histórico da média dos melhores locais preenchido, seguindo a mesma lógica.
        padded_historico_media_melhores_locais.append( 
            hm + [hm[-1]] * (max_len - len(hm)) if hm else [np.nan] * max_len 
        )
        # Adiciona o histórico do desvio padrão dos melhores locais preenchido, seguindo a mesma lógica.
        padded_historico_desvio_padrao_melhores_locais.append( 
            hdp + [hdp[-1]] * (max_len - len(hdp)) if hdp else [np.nan] * max_len 
        )

    # Calcula a média de todos os históricos de 'melhor_global' ao longo do eixo das iterações (axis=0).
    mean_historico_melhor_global = np.nanmean(padded_historico_melhor_global, axis=0) 
    # Calcula a média de todos os históricos de 'media_melhores_locais' ao longo do eixo das iterações.
    mean_historico_media_melhores_locais = np.nanmean(padded_historico_media_melhores_locais, axis=0) 
    # Calcula o desvio padrão de todos os históricos de 'media_melhores_locais' ao longo do eixo das iterações.
    std_historico_media_melhores_locais = np.nanstd(padded_historico_media_melhores_locais, axis=0) 

    # Cria um array de números para o eixo X, representando as iterações (de 1 até max_len).
    iteracoes = np.arange(1, max_len + 1) 

    # Define o estilo visual dos gráficos usando 'seaborn-v0_8-whitegrid', que oferece uma aparência limpa com grade.
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(12, 7)) # Cria uma nova figura e um conjunto de subplots (eixos) com um tamanho específico.

    # Plota a linha tracejada azul: Média do Melhor Z da Geração. 
    # Esta linha mostra a média dos melhores valores de aptidão encontrados *dentro de cada geração*.
    ax.plot(iteracoes, mean_historico_media_melhores_locais, 'b--', label=f'Média do Melhor Z da Geração ({algorithm_name})') 
    
    # Calcula o limite superior da área sombreada do desvio padrão (média + desvio padrão).
    upper_bound = mean_historico_media_melhores_locais + std_historico_media_melhores_locais 
    # Calcula o limite inferior da área sombreada do desvio padrão (média - desvio padrão).
    lower_bound = mean_historico_media_melhores_locais - std_historico_media_melhores_locais 
    # Preenche a área entre os limites superior e inferior com a cor azul e baixa opacidade, 
    # visualizando o desvio padrão da média do melhor Z da geração.
    ax.fill_between(iteracoes, lower_bound, upper_bound, color='blue', alpha=0.1, label='Desvio Padrão (±1σ)') 
    
    # Plota a linha sólida vermelha: Média do Melhor Valor Global Encontrado.
    # Esta linha mostra a média dos melhores valores de aptidão encontrados *acumulativamente* até cada geração.
    ax.plot(iteracoes, mean_historico_melhor_global, 'r-', label='Média do Melhor Valor Global Encontrado', linewidth=2) 

    # Define o título do gráfico.
    ax.set_title(f'Gráfico de Convergência - {algorithm_name} (Média de {num_runs} Execuções)', fontsize=16) 
    # Define o rótulo do eixo X.
    ax.set_xlabel('Número de Iterações/Gerações', fontsize=12) 
    # Define o rótulo do eixo Y.
    ax.set_ylabel('Valor da Função Objetivo (Z Ótimo)', fontsize=12) 

    # Adiciona a legenda ao gráfico no canto superior direito.
    ax.legend(loc='upper right', fontsize=10) 
    ax.grid(True) # Adiciona uma grade ao gráfico para facilitar a leitura.

    output_folder_analysis = "resultados_analise" # Define o nome da pasta onde os resultados gráficos serão salvos.
    os.makedirs(output_folder_analysis, exist_ok=True) # Cria o diretório de saída se ele não existir.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp formatado para garantir nomes de arquivos únicos.
    image_name = f"Grafico_Convergencia_{algorithm_name}_media_{num_runs}_runs_{timestamp}.png" # Monta o nome do arquivo da imagem.
    image_path = os.path.join(output_folder_analysis, image_name) # Combina o caminho da pasta e o nome do arquivo para criar o caminho completo.
    plt.savefig(image_path, dpi=300, bbox_inches='tight') # Salva o gráfico no caminho especificado com alta resolução e ajusta a caixa delimitadora.
    print(f"Gráfico de convergência salvo em: {image_path}") # Informa ao usuário onde o gráfico foi salvo.

    plt.show() # Exibe o gráfico na tela.


def plot_bar_graphs(results_df, algorithm_name="PSO", num_runs=1, pso_params_display=None):
    """
    Gera gráficos de barra para as estatísticas de avaliações e operações computacionais.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Define o estilo visual para os gráficos.

    # Calcula médias e desvios padrão para diversas métricas de desempenho a partir do DataFrame de resultados.
    mean_best_global = results_df['melhor_valor_global'].mean() # Média do melhor valor global encontrado.
    std_best_global = results_df['melhor_valor_global'].std() # Desvio padrão do melhor valor global encontrado.
    mean_iter_executed = results_df['iteracoes_executadas'].mean() # Média do número de iterações executadas.
    std_iter_executed = results_df['iteracoes_executadas'].std() # Desvio padrão do número de iterações executadas.

    mean_fo_evals_total = results_df['avaliacoes_funcao_total'].mean() # Média do total de avaliações da função objetivo.
    std_fo_evals_total = results_df['avaliacoes_funcao_total'].std() # Desvio padrão do total de avaliações da função objetivo.
    mean_mult_total = results_df['multiplicacoes_total'].mean() # Média do total de multiplicações.
    std_mult_total = results_df['multiplicacoes_total'].std() # Desvio padrão do total de multiplicações.
    mean_div_total = results_df['divisoes_total'].mean() # Média do total de divisões.
    std_div_total = results_df['divisoes_total'].std() # Desvio padrão do total de divisões.

    mean_fo_evals_min_global = results_df['avaliacoes_minimo_global'].mean() # Média de avaliações até o ponto do mínimo global.
    std_fo_evals_min_global = results_df['avaliacoes_minimo_global'].std() # Desvio padrão de avaliações até o ponto do mínimo global.
    mean_mult_min_global = results_df['multiplicacoes_minimo_global'].mean() # Média de multiplicações até o ponto do mínimo global.
    std_mult_min_global = results_df['multiplicacoes_minimo_global'].std() # Desvio padrão de multiplicações até o ponto do mínimo global.
    mean_div_min_global = results_df['divisoes_minimo_global'].mean() # Média de divisões até o ponto do mínimo global.
    std_div_min_global = results_df['divisoes_minimo_global'].std() # Desvio padrão de divisões até o ponto do mínimo global.

    # --- Criação dos Gráficos de Barra ---
    fig2, axes = plt.subplots(2, 2, figsize=(16, 9)) # Cria uma figura com uma grade de 2x2 subplots para os gráficos de barra.
    fig2.suptitle(f'Análise de Desempenho - {algorithm_name} (Média de {num_runs} Execuções)', fontsize=18) # Define um título superpuesto para toda a figura.

    # Gráfico 1: Melhor Valor Global e Iterações Executadas
    ax1 = axes[0, 0] # Seleciona o primeiro subplot (linha 0, coluna 0).
    labels1 = ['Melhor Valor Global', 'Iterações Executadas'] # Rótulos para as barras deste gráfico.
    values1 = [mean_best_global, mean_iter_executed] # Valores médios para as alturas das barras.
    errors1 = [std_best_global, std_iter_executed] # Valores de desvio padrão para as barras de erro.
    colors1 = ['skyblue', 'lightcoral'] # Cores para as barras.
    bars1 = ax1.bar(labels1, values1, yerr=errors1, capsize=5, color=colors1) # Cria o gráfico de barras com barras de erro e tampas.
    ax1.set_title('Média e Desvio Padrão (Melhor Valor e Iterações)', fontsize=14) # Define o título do subplot.
    ax1.set_ylabel('Valor / Iterações', fontsize=12) # Define o rótulo do eixo Y.
    # Adicionar os valores acima das barras
    for bar in bars1: # Itera sobre cada barra para adicionar seu valor numérico acima.
        yval = bar.get_height() # Obtém a altura da barra.
        # Adiciona o texto com o valor da barra, formatado para 2 casas decimais. Ajusta a posição vertical ligeiramente.
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval), 
                 f'{yval:.2f}', ha='center', va='bottom' if yval > 0 else 'top')

    # Gráfico 2: Avaliações e Operações - Total (para Convergência Completa do Algoritmo)
    ax2 = axes[0, 1] # Seleciona o segundo subplot (linha 0, coluna 1).
    labels2 = ['Avaliações Função', 'Multiplicações', 'Divisões'] # Rótulos para as barras.
    values2 = [mean_fo_evals_total, mean_mult_total, mean_div_total] # Valores médios.
    errors2 = [std_fo_evals_total, std_mult_total, std_div_total] # Desvios padrão.
    colors2 = ['lightgreen', 'salmon', 'plum'] # Cores.
    bars2 = ax2.bar(labels2, values2, yerr=errors2, capsize=5, color=colors2) # Cria o gráfico de barras.
    ax2.set_title('Média e Desvio Padrão (Avaliações e Operações - Total)', fontsize=14) # Título do subplot.
    ax2.set_ylabel('Quantidade', fontsize=12) # Rótulo do eixo Y.
    for bar in bars2: # Itera sobre as barras para adicionar valores numéricos.
        yval = bar.get_height()
        # Adiciona o texto com o valor da barra, formatado para 0 casas decimais (contagens).
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval), 
                 f'{yval:.0f}', ha='center', va='bottom' if yval > 0 else 'top')


    # Gráfico 3: Avaliações e Operações - Melhor Global (para o ponto onde o mínimo global foi encontrado)
    ax3 = axes[1, 0] # Seleciona o terceiro subplot (linha 1, coluna 0).
    labels3 = ['Avaliações', 'Multiplicações', 'Divisões'] # Rótulos para as barras.
    values3 = [mean_fo_evals_min_global, mean_mult_min_global, mean_div_min_global] # Valores médios.
    errors3 = [std_fo_evals_min_global, std_mult_min_global, std_div_min_global] # Desvios padrão.
    colors3 = ['orange', 'darkorange', 'mediumpurple'] # Cores.
    bars3 = ax3.bar(labels3, values3, yerr=errors3, capsize=5, color=colors3) # Cria o gráfico de barras.
    ax3.set_title('Média e Desvio Padrão (Avaliações e Operações - Melhor Global)', fontsize=14, pad=20) # Título do subplot.
    ax3.set_ylabel('Quantidade', fontsize=12) # Rótulo do eixo Y.
    for bar in bars3: # Itera sobre as barras para adicionar valores numéricos.
        yval = bar.get_height()
        # Adiciona o texto com o valor da barra, formatado para 0 casas decimais.
        ax3.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval), 
                 f'{yval:.0f}', ha='center', va='bottom' if yval > 0 else 'top')


    # Texto com Parâmetros da Simulação (no quarto subplot vazio)
    ax4 = axes[1, 1] # Seleciona o quarto subplot (linha 1, coluna 1).
    ax4.set_axis_off() # Desliga os eixos para este subplot, transformando-o em uma área de texto puro.

    if pso_params_display: # Verifica se o dicionário de parâmetros para exibição foi fornecido.
        params_text = ( # Formata uma string multi-linha com os parâmetros do PSO.
            f'Parâmetros da Simulação:\n'
            f'Função Otimizada: w29 + w1 (Minimização)\n'
            f'Limites da Função: {pso_params_display["limites_funcao"]}\n'
            f'Número de Partículas: {pso_params_display["num_particulas"]}\n'
            f'Número de Iterações Máximo: {pso_params_display["num_iteracoes"]}\n'
            f'Peso de Inércia (W_max): {pso_params_display["w_max"]:.2f}\n'
            f'Peso de Inércia (W_min): {pso_params_display["w_min"]:.2f}\n'
            f'Coeficiente Cognitivo (c1): {pso_params_display["c1"]:.0f}\n'
            f'Coeficiente Social (c2): {pso_params_display["c2"]:.0f}\n'
            f'Tolerância para Convergência: {pso_params_display["tolerancia"]}\n'
            f'Iterações sem Melhora Limite: {pso_params_display["iteracoes_sem_melhora_limite"]}\n'
        )
        # Adiciona o texto ao subplot na posição (0.5, 0.9) em coordenadas relativas aos eixos, 
        # centralizado e alinhado ao topo, com um bbox estilizado.
        ax4.text(0.5, 0.9, params_text, transform=ax4.transAxes, 
                 fontsize=11, verticalalignment='top', horizontalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout dos subplots para evitar sobreposição, com uma pequena margem na parte inferior.
    
    output_folder_analysis = "resultados_analise" # Define o diretório de saída para os gráficos.
    os.makedirs(output_folder_analysis, exist_ok=True) # Cria o diretório se não existir.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp.
    image_name = f"Graficos_Barras_{algorithm_name}_media_{num_runs}_runs_{timestamp}.png" # Monta o nome do arquivo da imagem.
    image_path = os.path.join(output_folder_analysis, image_name) # Cria o caminho completo para o arquivo.
    plt.savefig(image_path, dpi=300, bbox_inches='tight') # Salva a figura com alta resolução e ajuste de margens.
    print(f"Gráficos de barra salvos em: {image_path}") # Informa o usuário sobre o salvamento.

    plt.show() # Exibe a figura com os gráficos de barra.

# Função MODIFICADA para salvar o sumário em um arquivo CSV com UTF-8
def save_summary_to_csv(results_df, algorithm_name="PSO", num_runs=1, pso_params_display=None):
    """
    Salva as médias e desvios padrão das estatísticas de desempenho em um arquivo CSV.
    """
    output_folder_analysis = "resultados_analise" # Define o diretório de saída.
    os.makedirs(output_folder_analysis, exist_ok=True) # Cria o diretório se não existir.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp.
    file_name = f"Sumario_Desempenho_{algorithm_name}_media_{num_runs}_runs_{timestamp}.csv" # Monta o nome do arquivo CSV.
    file_path = os.path.join(output_folder_analysis, file_name) # Cria o caminho completo para o arquivo CSV.

    # Calcula médias e desvios padrão para as métricas de desempenho a partir do DataFrame.
    # Estas são as mesmas cálculos usados nos gráficos de barra.
    mean_best_global = results_df['melhor_valor_global'].mean() 
    std_best_global = results_df['melhor_valor_global'].std() 
    mean_iter_executed = results_df['iteracoes_executadas'].mean() 
    std_iter_executed = results_df['iteracoes_executadas'].std() 

    mean_fo_evals_total = results_df['avaliacoes_funcao_total'].mean() 
    std_fo_evals_total = results_df['avaliacoes_funcao_total'].std() 
    mean_mult_total = results_df['multiplicacoes_total'].mean() 
    std_mult_total = results_df['multiplicacoes_total'].std() 
    mean_div_total = results_df['divisoes_total'].mean() 
    std_div_total = results_df['divisoes_total'].std() 

    mean_fo_evals_min_global = results_df['avaliacoes_minimo_global'].mean() 
    std_fo_evals_min_global = results_df['avaliacoes_minimo_global'].std() 
    mean_mult_min_global = results_df['multiplicacoes_minimo_global'].mean() 
    std_mult_min_global = results_df['multiplicacoes_minimo_global'].std() 
    mean_div_min_global = results_df['divisoes_minimo_global'].mean() 
    std_div_min_global = results_df['divisoes_minimo_global'].std() 

    # Cria um DataFrame do Pandas para organizar os dados que serão salvos no CSV.
    data_to_csv = { 
        'Métrica': [ # Coluna com os nomes das métricas.
            'Melhor Valor Global Final',
            'Iterações Executadas (até convergência)',
            'Avaliações da Função Objetivo (Total)',
            'Multiplicações (Total)',
            'Divisões (Total)',
            'Avaliações da Função Objetivo (Mínimo Global)',
            'Multiplicações (Mínimo Global)',
            'Divisões (Mínimo Global)'
        ],
        'Media': [ # Coluna com as médias das métricas.
            mean_best_global,
            mean_iter_executed,
            mean_fo_evals_total,
            mean_mult_total,
            mean_div_total,
            mean_fo_evals_min_global,
            mean_mult_min_global,
            mean_div_min_global
        ],
        'Desvio_Padrao': [ # Coluna com os desvios padrão das métricas.
            std_best_global,
            std_iter_executed,
            std_fo_evals_total,
            std_mult_total,
            std_div_total,
            std_fo_evals_min_global,
            std_mult_min_global,
            std_div_min_global
        ]
    }
    summary_df = pd.DataFrame(data_to_csv) # Converte o dicionário em um DataFrame.

    # Adiciona os parâmetros de simulação como comentários no cabeçalho do arquivo CSV.
    header_comments = [ 
        f"# --- Sumário de Desempenho do Algoritmo {algorithm_name} ---", # Título do sumário.
        f"# Número de Execuções: {num_runs}", # Número de repetições da simulação.
        f"# Data e Hora da Análise: {timestamp}", # Data e hora da geração do arquivo.
        "#", # Linha em branco para formatação.
        "# Parâmetros da Simulação:" # Título para a seção de parâmetros.
    ]
    if pso_params_display: # Se os parâmetros para exibição existirem, itera sobre eles.
        for key, value in pso_params_display.items():
            # A função .title() pode introduzir caracteres especiais que precisam ser em UTF-8
            # Adiciona cada parâmetro como uma linha de comentário, formatando a chave.
            header_comments.append(f"#   {key.replace('_', ' ').title()}: {value}") 
    header_comments.append("#") # Adiciona uma linha em branco final para separação.

    # Salva o DataFrame no arquivo CSV, especificando encoding='utf-8'
    # Usa 'newline=""' para evitar problemas de linhas em branco extras no CSV
    # Abre o arquivo CSV no modo de escrita ('w') com a codificação 'utf-8' e 'newline=""' para tratamento de quebras de linha.
    # MODIFICADO AQUI: encoding='utf-8'
    with open(file_path, 'w', encoding='utf-8', newline='') as f: 
        for line in header_comments: # Escreve cada linha de comentário no arquivo.
            f.write(line + '\n')
        # Salva o DataFrame no arquivo. 'index=False' impede que o índice do DataFrame seja escrito.
        # 'float_format='%.4f'' formata os números de ponto flutuante com 4 casas decimais.
        # O argumento 'encoding' é tratado pela abertura do arquivo 'f' aqui, não precisa ser passado para to_csv.
        summary_df.to_csv(f, index=False, float_format='%.4f') 

    print(f"Sumário de desempenho salvo em: {file_path}") # Informa ao usuário onde o sumário foi salvo.


if __name__ == "__main__": # Este bloco de código só será executado quando o script for rodado diretamente (não quando importado como módulo).
    # --- VARIÁVEL PARA CONTROLAR QUANTAS VEZES O CÓDIGO RODA ---
    num_repeticoes_simulacao = 10 # Define o número de vezes que o algoritmo PSO será executado para coletar estatísticas.

    # --- Configuração dos Parâmetros do PSO ---
    # Define os parâmetros específicos do algoritmo PSO que serão utilizados em todas as simulações.
    limites_funcao = (-500, 500) # Define os limites do domínio da função objetivo (para as coordenadas x e y).
    num_particulas = 50 # Define o número de partículas (agentes) no enxame do PSO.
    num_iteracoes = 100 # Número MÁXIMO de iterações que cada execução do PSO pode rodar.
    w_max = 0.9 # Valor máximo do peso de inércia, geralmente usado no início para mais exploração.
    w_min = 0.4 # Valor mínimo do peso de inércia, usado no final para mais explotação.
    c1 = 3.5 # Coeficiente cognitivo, controla a influência do melhor histórico individual da partícula.
    c2 = 0.5 # Coeficiente social, controla a influência do melhor histórico global do enxame.
    tolerancia = 1e-6 # Um pequeno valor que define a tolerância para a convergência (quão pequena a mudança no melhor valor deve ser para parar).
    iteracoes_sem_melhora_limite = 20 # Número de iterações consecutivas sem melhora significativa que acionará o critério de parada.

    # Dicionário para passar os parâmetros para o display no gráfico de barras e TXT
    # Este dicionário agrupa os parâmetros do PSO para facilitar a passagem e exibição em outros lugares do script.
    pso_params_for_display = { 
        "limites_funcao": limites_funcao,
        "num_particulas": num_particulas,
        "num_iteracoes": num_iteracoes,
        "w_max": w_max,
        "w_min": w_min,
        "c1": c1,
        "c2": c2,
        "tolerancia": tolerancia,
        "iteracoes_sem_melhora_limite": iteracoes_sem_melhora_limite
    }

    # Listas para armazenar os históricos de cada execução (para gráfico de convergência)
    # Estas listas coletam os históricos de desempenho de CADA uma das 'num_repeticoes_simulacao'.
    all_historico_melhor_global = [] 
    all_historico_media_melhores_locais = [] 
    all_historico_desvio_padrao_melhores_locais = [] 

    # Lista para armazenar as estatísticas finais de cada execução (para gráficos de barra e CSV)
    # Esta lista coletará os resultados finais de cada execução (e.g., melhor valor final, número de iterações).
    all_final_stats = [] 

    print(f"\n--- Iniciando {num_repeticoes_simulacao} execuções do PSO ---") # Informa o início das simulações.

    # Loop principal que executa o PSO várias vezes.
    for i in range(num_repeticoes_simulacao): 
        print(f"\nExecução {i+1}/{num_repeticoes_simulacao}:") # Informa o progresso das execuções.
        # Instancia a classe PSO com a função objetivo e os parâmetros definidos.
        otimizador_pso = PSO(minha_funcao_w29_w1, limites_funcao, num_particulas, num_iteracoes, 
                             w_max, w_min, c1, c2, tolerancia, iteracoes_sem_melhora_limite)

        pso_results = otimizador_pso.executar() # Executa o algoritmo PSO e armazena o dicionário de resultados.

        # Armazena os históricos de convergência desta execução específica nas listas coletoras.
        all_historico_melhor_global.append(pso_results["historico_melhor_global"]) 
        all_historico_media_melhores_locais.append(pso_results["historico_media_melhores_locais"]) 
        all_historico_desvio_padrao_melhores_locais.append(pso_results["historico_desvio_padrao_melhores_locais"]) 

        # Armazena as estatísticas finais desta execução específica na lista 'all_final_stats'.
        all_final_stats.append({ 
            'melhor_valor_global': pso_results['melhor_valor_global'],
            'iteracoes_executadas': pso_results['iteracoes_executadas'],
            'avaliacoes_funcao_total': pso_results['avaliacoes_funcao_total'],
            'multiplicacoes_total': pso_results['multiplicacoes_total'],
            'divisoes_total': pso_results['divisoes_total'],
            'avaliacoes_minimo_global': pso_results['avaliacoes_minimo_global'],
            'multiplicacoes_minimo_global': pso_results['multiplicacoes_minimo_global'],
            'divisoes_minimo_global': pso_results['divisoes_minimo_global'],
        })

    print(f"\n--- {num_repeticoes_simulacao} execuções concluídas. Gerando gráficos e sumário ---") # Informa a conclusão das execuções e o início da geração de relatórios.

    # Converter a lista de dicionários para um DataFrame do pandas para facilitar cálculos estatísticos
    # Converte a lista de dicionários contendo as estatísticas finais de cada execução em um DataFrame do Pandas,
    # o que simplifica o cálculo de médias e desvios padrão para os gráficos e o sumário CSV.
    results_df = pd.DataFrame(all_final_stats) 

    # 1. Gerar o Gráfico de Convergência (gráfico de linha)
    # Chama a função para plotar o gráfico de convergência médio das múltiplas execuções.
    plot_convergence_graph( 
        all_historico_melhor_global,
        all_historico_media_melhores_locais,
        all_historico_desvio_padrao_melhores_locais,
        algorithm_name="PSO", # Passa o nome do algoritmo para o título do gráfico.
        num_runs=num_repeticoes_simulacao # Passa o número de execuções para a função de plotagem.
    )

    # 2. Gerar os Gráficos de Barra
    # Chama a função para plotar os gráficos de barra que sumarizam o desempenho computacional e de resultado.
    plot_bar_graphs(results_df, algorithm_name="PSO", num_runs=num_repeticoes_simulacao, 
                    pso_params_display=pso_params_for_display)

    # 3. NOVO: Salvar o sumário em CSV
    # Chama a função para salvar as estatísticas sumarizadas (médias e desvios padrão) em um arquivo CSV.
    save_summary_to_csv(results_df, algorithm_name="PSO", num_runs=num_repeticoes_simulacao, 
                        pso_params_display=pso_params_for_display)

    print("\nAnálise completa. Verifique a pasta 'resultados_analise' para os gráficos e o sumário.") # Mensagem final para o usuário.