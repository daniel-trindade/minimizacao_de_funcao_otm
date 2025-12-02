# Grafico.py
# Importa as bibliotecas necessárias para plotagem
import numpy as np # NumPy para operações numéricas, especialmente para meshgrid e arrays
import matplotlib.pyplot as plt # Matplotlib para criação e exibição de gráficos
from mpl_toolkits.mplot3d import Axes3D # Para gráficos 3D (necessário para 'projection="3d"')

# Importa a função objetivo a ser visualizada no gráfico
from funcoes_otimizacao import minha_funcao_w29_w1 

def GraficoPSO(enxame, iteracao, ax, melhor_valor_global, pso_params=None):
    """
    Função para plotar o gráfico do PSO (Particle Swarm Optimization) em 3D.
    Exibe a superfície da função objetivo e a posição das partículas.
    
    Parâmetros:
        enxame (list): Lista de objetos Particula, representando o enxame.
        iteracao (int): O número da iteração atual do PSO.
        ax (matplotlib.axes._subplots.Axes3DSubplot): O objeto Axes3D onde o gráfico será plotado.
        melhor_valor_global (float): O melhor valor da função objetivo encontrado globalmente até o momento.
        pso_params (dict, optional): Um dicionário contendo parâmetros e estatísticas do PSO para exibição na legenda.
    """
    ax.clear() # Limpa o Axes para redesenhar o gráfico a cada iteração

    # Define a grade para plotar a superfície da função objetivo
    x_grid = np.linspace(-500, 500, 100) # 100 pontos no eixo X de -500 a 500
    y_grid = np.linspace(-500, 500, 100) # 100 pontos no eixo Y de -500 a 500
    X, Y = np.meshgrid(x_grid, y_grid) # Cria uma grade 2D a partir dos vetores x_grid e y_grid
    Z = minha_funcao_w29_w1(X, Y) # Calcula os valores Z da função objetivo para cada ponto da grade

    # Define o título do gráfico, incluindo a iteração atual e o melhor valor global
    ax.set_title(f'PSO - Iteração {iteracao} | Melhor Valor: {melhor_valor_global:.4f}', fontsize=12)
    ax.set_xlabel('x') # Rótulo para o eixo X
    ax.set_ylabel('y') # Rótulo para o eixo Y
    ax.set_zlabel('F(x, y)') # Rótulo para o eixo Z (valor da função)

    # Define os limites dos eixos para garantir uma visualização consistente
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])

    # Plota a superfície da função objetivo
    # rstride e cstride controlam a densidade das linhas na superfície
    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.4, rstride=5, cstride=5)

    # Cores para diferenciar as partículas (reutiliza cores se houver mais partículas que cores)
    colors = ['red', 'yellow', 'green', 'blue', 'pink', 'purple', 'orange', 'black', 'gray', 'brown']
    for idx, particula in enumerate(enxame): # Itera sobre cada partícula no enxame
        particula_x = particula.posicao_i[0] # Posição X da partícula
        particula_y = particula.posicao_i[1] # Posição Y da partícula
        # Calcula o valor Z da função objetivo na posição atual da partícula
        particle_z_val = minha_funcao_w29_w1(np.array(particula_x), np.array(particula_y))
        color = colors[idx % len(colors)] # Seleciona uma cor da lista
        # Plota a partícula como um ponto 3D
        ax.scatter(particula_x, particula_y, particle_z_val, color=color, s=100, edgecolors='black', linewidth=0.5)

    # Define os limites do eixo Z para garantir que todas as partículas e a superfície sejam visíveis
    ax.set_zlim([-600, 4000])

    # Exibe os parâmetros e estatísticas do PSO na legenda se o dicionário pso_params for fornecido
    if pso_params:
        # Formata o texto da legenda com todos os parâmetros e estatísticas
        legend_text = (
            f'--- Parâmetros PSO ---\n'
            f'C1: {pso_params["c1"]:.2f}\n'
            f'C2: {pso_params["c2"]:.2f}\n'
            f'W_max: {pso_params["w_max"]:.2f}\n'
            f'W_min: {pso_params["w_min"]:.2f}\n'
            f'Partículas: {pso_params["num_particulas"]}\n'
            f'Iterações Totais: {pso_params["num_iteracoes_max"]}\n' # CHAVE CORRIGIDA AQUI
            f'Limite It. sem melhora: {pso_params["limite_iteracoes_sem_melhora"]}\n\n'
            f'--- Estatísticas de Convergência ---\n'
            f'Avaliações FO: {pso_params["avaliacoes_funcao"]}\n'
            f'Mult: {pso_params["multiplicacoes_total"]}\n'
            f'Div: {pso_params["divisoes_total"]}\n'
            f'Iterações sem melhora: {pso_params["iteracoes_sem_melhora"]}\n'
            f'--- Estatísticas no Melhor Global ---\n'
            f'Avaliações: {pso_params["avaliacoes_minimo_global"]}\n'
            f'Mult: {pso_params["multiplicacoes_minimo_global"]}\n'
            f'Div: {pso_params["divisoes_minimo_global"]}'
        )
        
        ax.text2D(1.1, 0.98, legend_text, 
                  transform=ax.transAxes, # Transforma as coordenadas para serem relativas aos eixos
                  fontsize=10, 
                  verticalalignment='top', 
                  horizontalalignment='left',
                  bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5))

    plt.pause(0.01) # Pequena pausa para permitir que o gráfico seja atualizado visualmente

def GraficoAG(populacao, melhor_solucao, iteracao, ax, melhor_aptidao_global, ag_params=None):
    """
    Função para plotar o gráfico do Algoritmo Genético (AG) em 3D.
    Exibe a superfície da função objetivo e a posição dos indivíduos da população.
    
    Parâmetros:
        populacao (list): Lista de arrays NumPy, representando os indivíduos da população.
        melhor_solucao (np.array): A melhor solução (indivíduo) encontrada globalmente até o momento.
        iteracao (int): O número da geração atual do AG.
        ax (matplotlib.axes._subplots.Axes3DSubplot): O objeto Axes3D onde o gráfico será plotado.
        melhor_aptidao_global (float): A melhor aptidão (valor da função objetivo) global encontrada até o momento.
        ag_params (dict, optional): Um dicionário contendo parâmetros e estatísticas do AG para exibição na legenda.
    """
    ax.clear() # Limpa o Axes para redesenhar o gráfico a cada iteração

    # Define a grade para plotar a superfície da função objetivo
    x_grid = np.linspace(-500, 500, 100) # 100 pontos no eixo X de -500 a 500
    y_grid = np.linspace(-500, 500, 100) # 100 pontos no eixo Y de -500 a 500
    X, Y = np.meshgrid(x_grid, y_grid) # Cria uma grade 2D a partir dos vetores x_grid e y_grid
    Z = minha_funcao_w29_w1(X, Y) # Calcula os valores Z da função objetivo para cada ponto da grade

    # Define o título do gráfico, incluindo a geração atual e o melhor valor global
    ax.set_title(f'AG - Geração {iteracao} | Melhor Valor: {melhor_aptidao_global:.4f}')
    ax.set_xlabel('x') # Rótulo para o eixo X
    ax.set_ylabel('y') # Rótulo para o eixo Y
    ax.set_zlabel('F(x, y)') # Rótulo para o eixo Z (valor da função)

    # Define os limites dos eixos para garantir uma visualização consistente
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])

    # Plota a superfície da função objetivo
    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.5, rstride=5, cstride=5)

    population_color = 'blue' # Cor para os indivíduos da população
    best_solution_color = 'red' # Cor para a melhor solução global

    for individuo in populacao: # Itera sobre cada indivíduo na população
        ind_x, ind_y = individuo[0], individuo[1] # Posição X e Y do indivíduo
        # Calcula o valor Z da função objetivo na posição atual do indivíduo
        ind_z_val = minha_funcao_w29_w1(np.array(ind_x), np.array(ind_y))
        # Plota o indivíduo como um ponto 3D
        ax.scatter(ind_x, ind_y, ind_z_val, color=population_color, s=50, alpha=0.7)

    # Se uma melhor solução global foi encontrada, plota-a destacadamente
    if melhor_solucao is not None:
        best_x, best_y = melhor_solucao[0], melhor_solucao[1] # Posição X e Y da melhor solução
        # Calcula o valor Z da função objetivo na posição da melhor solução
        best_z_val = minha_funcao_w29_w1(np.array(best_x), np.array(best_y))
        # Plota a melhor solução como um ponto maior e de cor diferente
        ax.scatter(best_x, best_y, best_z_val, color=best_solution_color, s=200, marker='o', edgecolor='black', linewidth=1.5, label='Melhor Solução Global')

    # Define os limites do eixo Z para garantir que todos os indivíduos e a superfície sejam visíveis
    ax.set_zlim([-500, 4000])

    # Exibe os parâmetros e estatísticas do AG na legenda se o dicionário ag_params for fornecido
    if ag_params:
        # Formata o texto da legenda com todos os parâmetros e estatísticas
        legend_text = (
            f'--- Parâmetros AG ---\n'
            f'Tamanho Pop: {ag_params["tamanho_populacao"]}\n'
            f'Taxa Mutação: {ag_params["taxa_mutacao"]:.2f}\n'
            f'Taxa Crossover: {ag_params["taxa_crossover"]:.2f}\n'
            f'Iterações Totais: {ag_params["iteracoes_totais"]}\n'
            f'Limite Ger. sem melhora: {ag_params["limite_geracoes_sem_melhora"]}\n\n'
            f'--- Estatísticas de Convergência ---\n'
            f'Avaliações FO: {ag_params["avaliacoes_funcao"]}\n'
            f'Mult: {ag_params["multiplicacoes_total"]}\n'
            f'Div: {ag_params["divisoes_total"]}\n'
            f'Gerações sem melhora: {ag_params["geracoes_sem_melhora"]}\n'
            f'--- Estatísticas no Melhor Global ---\n'
            f'Avaliações: {ag_params["avaliacoes_minimo_global"]}\n'
            f'Mult: {ag_params["multiplicacoes_minimo_global"]}\n'
            f'Div: {ag_params["divisoes_minimo_global"]}'
        )
        
        # Adiciona o texto da legenda ao gráfico em 2D
        ax.text2D(1.1, 0.98, legend_text, 
                  transform=ax.transAxes, # Transforma as coordenadas para serem relativas aos eixos
                  fontsize=10, 
                  verticalalignment='top', 
                  horizontalalignment='left',
                  bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5))

    plt.pause(0.01) # Pequena pausa para permitir que o gráfico seja atualizado visualmente