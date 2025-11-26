# PSO.py

# Importando bibliotecas padrão
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# impotando módulos personalizados
from funcoes_otimizacao import minha_funcao_w29_w1 # A função objetivo W29 + W1 a ser minimizada
from Grafico import GraficoPSO # Importa a função para plotar o gráfico do PSO (visualização 3D)
from utils import global_op_counter, FuncaoObjetivoWrapper # Contadores globais e o wrapper para a função objetivo

# --- Definição da Classe Particula ---
class Particula:
    """
    Representa uma única partícula no algoritmo de Otimização por Enxame de Partículas (PSO).
    Cada partícula possui uma posição, velocidade e a melhor posição já encontrada por ela.
    """
    def __init__(self, limites):
        """
        Construtor da classe Particula.
        Inicializa a partícula com uma posição e velocidade aleatórias dentro dos limites definidos.
        """
        # A posição da partícula é um array NumPy de 2 dimensões (x, y) gerado aleatoriamente.
        self.posicao_i = np.array([random.uniform(limites[0], limites[1]),
                                     random.uniform(limites[0], limites[1])])
        
        # A velocidade da partícula é um array NumPy de 2 dimensões (vx, vy) gerado aleatoriamente.
        # A velocidade inicial é geralmente menor que a faixa dos limites de posição.
        self.velocidade_i = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        
        # Melhor posição que esta partícula já encontrou (inicialmente é a posição atual).
        self.melhor_posicao_i = self.posicao_i.copy()
        
        # O valor da função objetivo na melhor_posicao_i (inicialmente infinito, pois queremos minimizar).
        self.melhor_valor_i = float('inf')



# --- Definição da Classe PSO (Particle Swarm Optimization) ---
class PSO:
    """
    Implementa o algoritmo de Otimização por Enxame de Partículas (PSO).
    Gerencia o enxame de partículas, atualiza suas posições e velocidades,
    e busca o mínimo global da função objetivo.
    """
    def __init__(self, funcao_objetivo, limites, num_particulas, num_iteracoes, w_max, w_min, c1, c2, tolerancia=1e-6, iteracoes_sem_melhora_limite=50):
        """
        Construtor da classe PSO.
        
        Parâmetros:
            funcao_objetivo (callable): A função a ser otimizada (minimizada).
            limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca.
            num_particulas (int): O número de partículas no enxame.
            num_iteracoes (int): O número máximo de iterações do algoritmo.
            w_max (float): Coeficiente de inércia máximo.
            w_min (float): Coeficiente de inércia mínimo.
            c1 (float): Coeficiente de aceleração cognitiva (influência da melhor posição da própria partícula).
            c2 (float): Coeficiente de aceleração social (influência da melhor posição global do enxame).
            tolerancia (float): Tolerância para o critério de parada por convergência.
            iteracoes_sem_melhora_limite (int): Número de iterações sem melhora significativa para parar o algoritmo.
        """
        self.funcao_objetivo = funcao_objetivo # Função objetivo (w29_w1, por exemplo)
        self.limites = limites # Limites do espaço de busca (ex: (-500, 500))
        self.num_particulas = num_particulas # Número de partículas
        self.num_iteracoes = num_iteracoes # Número máximo de iterações
        self.w_max = w_max # Coeficiente de inércia (máximo, para inércia decrescente)
        self.w_min = w_min # Coeficiente de inércia mínimo
        self.c1 = c1 # Coeficiente cognitivo
        self.c2 = c2 # Coeficiente social
        self.tolerancia = tolerancia # Tolerância para convergência
        self.iteracoes_sem_melhora_limite = iteracoes_sem_melhora_limite # Limite para parada por falta de melhora
        
        # Inicialização da melhor posição global e seu valor
        self.melhor_posicao_global = None # A melhor posição já encontrada por qualquer partícula no enxame
        self.melhor_valor_global = float('inf') # O valor da função objetivo na melhor_posicao_global (inicialmente infinito)

        # Contadores de estatísticas para o melhor global
        self.avaliacoes_pso_minimo_global = 0 # Avaliações da função quando o melhor global foi encontrado
        self.operacoes_pso_minimo_global_mult = 0 # Multiplicações quando o melhor global foi encontrado
        self.operacoes_pso_minimo_global_div = 0 # Divisões quando o melhor global foi encontrado

        # Reseta os contadores globais de operações no início da execução do PSO
        global_op_counter.reset()
        # Cria um "wrapper" para a função objetivo que também conta suas avaliações
        self.minha_funcao_w29_w1_wrapper = FuncaoObjetivoWrapper(self.funcao_objetivo, global_op_counter)

        # --- NOVAS VARIÁVEIS PARA ARMAZENAR DADOS DE CONVERGÊNCIA ---
        self.historico_melhor_global = [] # Armazena o melhor valor global a cada iteração
        self.historico_media_melhores_locais = [] # Armazena a média dos melhores valores locais das partículas a cada iteração
        self.historico_desvio_padrao_melhores_locais = [] # Armazena o desvio padrão dos melhores valores locais das partículas a cada iteração


    def inicializar_enxame(self):
        """
        Cria e inicializa todas as partículas no enxame.
        """
        enxame = [] # Lista para armazenar as partículas
        for _ in range(self.num_particulas): # Loop para criar o número especificado de partículas
            enxame.append(Particula(self.limites)) # Adiciona uma nova partícula à lista
        return enxame # Retorna o enxame inicializado

    def executar(self):
        """
        Executa o algoritmo PSO principal.
        Gera o enxame, itera, atualiza as partículas e encontra o mínimo.
        
        Retorna:
            dict: Um dicionário contendo as estatísticas finais da execução do PSO.
        """
        enxame = self.inicializar_enxame() # Inicializa o enxame de partículas
        
        iteracoes_sem_melhora = 0 # Contador para iterações consecutivas sem melhora significativa
        ultima_melhor_valor_global = float('inf') # Armazena o último melhor valor global para o critério de convergência
        
        # --- INICIALIZAÇÃO DO GRÁFICO PARA PSO (Visualização 3D em tempo real) ---
        # Abertura da figura e do eixo 3D apenas se for realmente para plotar em tempo real.
        # Se você não quer a visualização 3D em tempo real, pode comentar ou remover estas linhas.
        fig = plt.figure(figsize=(11, 8)) # Cria uma nova figura para o gráfico
        ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura
        plt.subplots_adjust(right=0.7) # Ajuste de layout

        i = 0 # Contador de iterações
        while i < self.num_iteracoes: # Loop principal do algoritmo PSO, uma iteração por vez
            # Atualiza o coeficiente de inércia 'w' linearmente decrescente ao longo das iterações
            # Isso ajuda a equilibrar exploração (w alto no início) e explotação (w baixo no final)
            self.w = self.w_max - (self.w_max - self.w_min) * i / self.num_iteracoes
            global_op_counter.add_mult(1) # Contabiliza a multiplicação na fórmula do 'w'
            global_op_counter.add_div(1) # Contabiliza a divisão na fórmula do 'w'

            # --- Coleta dos melhores valores locais para calcular média e desvio padrão da geração ---
            melhores_valores_locais_geracao = []

            for particula in enxame: # Itera sobre cada partícula no enxame
                # Avalia a função objetivo na posição atual da partícula
                valor_atual = self.minha_funcao_w29_w1_wrapper(particula.posicao_i[0], particula.posicao_i[1])
                
                # Atualiza a melhor posição local da partícula se o valor atual for melhor
                if valor_atual < particula.melhor_valor_i:
                    particula.melhor_valor_i = valor_atual # Atualiza o melhor valor local
                    particula.melhor_posicao_i = particula.posicao_i.copy() # Atualiza a melhor posição local

                # Atualiza a melhor posição global do enxame se a melhor posição local da partícula for melhor
                if valor_atual < self.melhor_valor_global:
                    self.melhor_valor_global = valor_atual # Atualiza o melhor valor global
                    self.melhor_posicao_global = particula.posicao_i.copy() # Atualiza a melhor posição global
                    
                    # Registra as avaliações e operações *no momento* em que o melhor global foi encontrado
                    self.avaliacoes_pso_minimo_global = self.minha_funcao_w29_w1_wrapper.evaluations
                    self.operacoes_pso_minimo_global_mult = global_op_counter.multiplications
                    self.operacoes_pso_minimo_global_div = global_op_counter.divisions
                
                melhores_valores_locais_geracao.append(particula.melhor_valor_i) # Coleta o melhor valor local da partícula

            # --- Armazenamento dos dados de convergência para o histórico ---
            self.historico_melhor_global.append(self.melhor_valor_global)
            self.historico_media_melhores_locais.append(np.mean(melhores_valores_locais_geracao))
            self.historico_desvio_padrao_melhores_locais.append(np.std(melhores_valores_locais_geracao))


            # --- Lógica de parada por convergência (se a melhora global for menor que a tolerância) ---
            if i > 0 and abs(self.melhor_valor_global - ultima_melhor_valor_global) < self.tolerancia:
                iteracoes_sem_melhora += 1 # Incrementa o contador de iterações sem melhora
            else:
                iteracoes_sem_melhora = 0 # Reseta o contador se houve melhora
                
            ultima_melhor_valor_global = self.melhor_valor_global # Atualiza o último melhor valor global para a próxima comparação

            if iteracoes_sem_melhora >= self.iteracoes_sem_melhora_limite: # Verifica se o limite foi atingido
                print(f"\n[PSO] Parada por convergência: Mudança no melhor valor global menor que {self.tolerancia} por {self.iteracoes_sem_melhora_limite} iterações.")
                break # Sai do loop principal se houver convergência

            # Atualiza a velocidade e posição de cada partícula
            for particula in enxame:
                # Gerar números aleatórios r1 e r2 para os componentes cognitivo e social
                r1 = random.random()
                r2 = random.random()
                
                # Componente cognitivo: puxa a partícula para sua melhor posição local
                # C1 * r1 * (p_best - current_pos)
                cognitive_component = self.c1 * r1 * (particula.melhor_posicao_i - particula.posicao_i)
                global_op_counter.add_mult(2) # Contabiliza 2 multiplicações (uma para cada dimensão x,y)
                
                # Componente social: puxa a partícula para a melhor posição global do enxame
                # C2 * r2 * (g_best - current_pos)
                social_component = self.c2 * r2 * (self.melhor_posicao_global - particula.posicao_i)
                global_op_counter.add_mult(2) # Contabiliza 2 multiplicações (uma para cada dimensão x,y)

                # Atualiza a velocidade da partícula
                # nova_velocidade = w * velocidade_atual + componente_cognitivo + componente_social
                particula.velocidade_i = (self.w * particula.velocidade_i +
                                         cognitive_component +
                                         social_component)
                global_op_counter.add_mult(2) # Contabiliza 2 multiplicações (w * velocidade_atual para cada dimensão)

                # Limita a velocidade para evitar que as partículas "voem" muito longe
                particula.velocidade_i = np.clip(particula.velocidade_i, -self.limites[1], self.limites[1])
                
                # Atualiza a posição da partícula
                # nova_posicao = posicao_atual + nova_velocidade
                particula.posicao_i = particula.posicao_i + particula.velocidade_i
                
                # Garante que a partícula permaneça dentro dos limites do espaço de busca
                particula.posicao_i = np.clip(particula.posicao_i, self.limites[0], self.limites[1])
            
            # --- Dicionário com parâmetros e estatísticas para passar ao gráfico 3D (se ativo) ---
            pso_params_for_plot = {
                "c1": self.c1,
                "c2": self.c2,
                "w_max": self.w_max,
                "w_min": self.w_min,
                "num_iteracoes_max": self.num_iteracoes,
                "num_particulas": self.num_particulas,
                "avaliacoes_funcao": self.minha_funcao_w29_w1_wrapper.evaluations,
                "multiplicacoes_total": global_op_counter.multiplications,
                "divisoes_total": global_op_counter.divisions,
                "avaliacoes_minimo_global": self.avaliacoes_pso_minimo_global,
                "multiplicacoes_minimo_global": self.operacoes_pso_minimo_global_mult,
                "divisoes_minimo_global": self.operacoes_pso_minimo_global_div,
                "iteracoes_sem_melhora": iteracoes_sem_melhora,
                "limite_iteracoes_sem_melhora": self.iteracoes_sem_melhora_limite,
                "melhor_posicao_global": self.melhor_posicao_global
            }

            # Chama a função para desenhar o gráfico 3D da iteração atual
            GraficoPSO(enxame, i+1, ax, self.melhor_valor_global, pso_params=pso_params_for_plot) 
            i += 1 # Incrementa o contador de iterações.

        # --- Salva a imagem do gráfico 3D final ---
        output_folder_images = "resultados_pso_graficos" # Define a pasta para salvar as imagens
        os.makedirs(output_folder_images, exist_ok=True) # Cria a pasta se ela não existir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp para o nome do arquivo
        # Nome do arquivo inclui o timestamp e o melhor valor global para fácil identificação
        image_name = f"PSO_final_plot_{timestamp}_Valor_{self.melhor_valor_global:.4f}.png"
        image_path = os.path.join(output_folder_images, image_name)

        plt.savefig(image_path, dpi=300, bbox_inches='tight') # Salva a figura atual com alta resolução
        print(f"Gráfico final do PSO (3D) salvo em: {image_path}")
        
        plt.close(fig) # Fecha a janela do gráfico para liberar recursos

        # --- Prepara as estatísticas para impressão e salvamento ---
        stats_output = [] # Lista para armazenar as linhas de saída
        stats_output.append("--- Resultados Finais do Algoritmo PSO ---")
        stats_output.append(f"Função Otimizada: w29 + w1 (Minimização)")
        stats_output.append(f"Parâmetros do Algoritmo:")
        stats_output.append(f"   Limites da Função: {self.limites}")
        stats_output.append(f"   Número de Partículas: {self.num_particulas}")
        stats_output.append(f"   Número de Iterações Máximo: {self.num_iteracoes}")
        stats_output.append(f"   Peso de Inércia (W_max): {self.w_max}")
        stats_output.append(f"   Peso de Inércia (W_min): {self.w_min}")
        stats_output.append(f"   Coeficiente Cognitivo (c1): {self.c1}")
        stats_output.append(f"   Coeficiente Social (c2): {self.c2}")
        stats_output.append(f"   Tolerância para Convergência: {self.tolerancia}")
        stats_output.append(f"   Iterações sem Melhora Limite: {self.iteracoes_sem_melhora_limite}")
        stats_output.append("--------------------------------------------")
        stats_output.append(f"Melhor solução encontrada (PSO): {self.melhor_posicao_global}")
        stats_output.append(f"Valor da função para a melhor solução (PSO): {self.melhor_valor_global:.4f}")
        stats_output.append(f"Iterações executadas (PSO): {i}")
        stats_output.append(f"Avaliações da função objetivo (Total): {self.minha_funcao_w29_w1_wrapper.evaluations}")
        stats_output.append(f"Operações de Multiplicação (Total): {global_op_counter.multiplications}")
        stats_output.append(f"Operações de Divisão (Total): {global_op_counter.divisions}")
        stats_output.append(f"Avaliações para o 'melhor global' (momento de encontro): {self.avaliacoes_pso_minimo_global}")
        stats_output.append(f"Multiplicações para o 'melhor global' (momento de encontro): {self.operacoes_pso_minimo_global_mult}")
        stats_output.append(f"Divisões para o 'melhor global' (momento de encontro): {self.operacoes_pso_minimo_global_div}")
        stats_output.append("--------------------------------------------")

        for line in stats_output:
            print(line)

        # Retorna um dicionário com todas as estatísticas numéricas para análise externa,
        # incluindo os históricos de convergência
        results = {
            "melhor_posicao_global": self.melhor_posicao_global,
            "melhor_valor_global": self.melhor_valor_global,
            "iteracoes_executadas": i,
            "avaliacoes_funcao_total": self.minha_funcao_w29_w1_wrapper.evaluations,
            "multiplicacoes_total": global_op_counter.multiplications,
            "divisoes_total": global_op_counter.divisions,
            "avaliacoes_minimo_global": self.avaliacoes_pso_minimo_global,
            "multiplicacoes_minimo_global": self.operacoes_pso_minimo_global_mult,
            "divisoes_minimo_global": self.operacoes_pso_minimo_global_div,
            "limites_funcao": self.limites,
            "num_particulas": self.num_particulas,
            "num_iteracoes_max": self.num_iteracoes,
            "w_max": self.w_max,
            "w_min": self.w_min,
            "c1": self.c1,
            "c2": self.c2,
            "tolerancia": self.tolerancia,
            "iteracoes_sem_melhora_limite": self.iteracoes_sem_melhora_limite,
            "historico_melhor_global": self.historico_melhor_global, # NOVO
            "historico_media_melhores_locais": self.historico_media_melhores_locais, # NOVO
            "historico_desvio_padrao_melhores_locais": self.historico_desvio_padrao_melhores_locais # NOVO
        }
        return results

# Este bloco só será executado se PSO.py for o script principal rodado (ex: python PSO.py)
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo PSO (se executado como script principal) ---")
    # Define os parâmetros de teste para uma execução direta
    limites_funcao = (-500, 500) # Limites para a função w29+w1
    num_particulas = 15 # Número de partículas
    num_iteracoes = 70 # Número de iterações
    w_max = 0.7 # Valor máximo para o peso de inércia 'w'
    w_min = 0.2 # Valor minimo para o peso de inércia 'w'
    c1 = 2 # Coeficiente cognitivo
    c2 = 2 # Coeficiente social
    tolerancia = 1e-6 # Tolerância para critério de parada
    iteracoes_sem_melhora_limite = 20 # Limite de iterações sem melhora para parar

    # Cria uma instância do otimizador PSO
    otimizador_pso = PSO(minha_funcao_w29_w1, limites_funcao, num_particulas, num_iteracoes, w_max, w_min, c1, c2, tolerancia, iteracoes_sem_melhora_limite)
    
    # Executa o algoritmo PSO e captura os resultados
    final_results = otimizador_pso.executar()
    
    # Você pode acessar os resultados aqui, por exemplo:
    # print(f"Melhor Valor Global Capturado: {final_results['melhor_valor_global']:.4f}")
    # print(f"Iterações Executadas Capturadas: {final_results['iteracoes_executadas']}")