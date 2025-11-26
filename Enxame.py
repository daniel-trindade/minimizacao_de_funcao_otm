import random # Importa a biblioteca 'random' para gerar números aleatórios, essenciais para a inicialização e atualização das partículas.
import numpy as np # Importa a biblioteca 'numpy' como 'np', fundamental para operações com arrays numéricos de forma eficiente (vetorização).
from utils import global_op_counter # Importa 'global_op_counter' do módulo 'utils', que é um objeto global para contar operações aritméticas (multiplicações, divisões, adições/subtrações).

class Enxame: # Define a classe 'Enxame', que representa uma única partícula dentro do algoritmo de Otimização por Enxame de Partículas (PSO).
    def __init__(self, limites, c1, c2, w_max, w_min): # Define o método construtor da classe 'Enxame'. Ele é chamado ao criar uma nova partícula.
        # Inicializa a posição atual da partícula como um array NumPy.
        # random.uniform(a, b) gera um número de ponto flutuante aleatório entre 'a' e 'b'.
        # 'limites[0][0]' e 'limites[0][1]' são os limites mínimo e máximo para a coordenada x.
        # 'limites[1][0]' e 'limites[1][1]' são os limites mínimo e máximo para a coordenada y.
        self.posicao_i = np.array([random.uniform(limites[0][0], limites[0][1]),
                                     random.uniform(limites[1][0], limites[1][1])])
        
        # Inicializa a velocidade atual da partícula como um array NumPy.
        # As velocidades são iniciadas aleatoriamente entre -1 e 1.
        self.velocidade_i = np.array([random.uniform(-1, 1), 
                                       random.uniform(-1, 1)])
        
        # 'melhor_posicao_i' armazena a melhor posição (com o menor valor da função objetivo)
        # que esta partícula encontrou até o momento. É inicializada com a posição atual.
        self.melhor_posicao_i = np.array(self.posicao_i)
        # 'melhor_valor_i' armazena o valor da função objetivo para 'melhor_posicao_i'.
        # Inicializado com infinito para garantir que qualquer valor real seja considerado melhor.
        self.melhor_valor_i = float('inf')
        # 'valor_atual_i' armazena o valor da função objetivo para a 'posicao_i' atual.
        # Inicializado com infinito.
        self.valor_atual_i = float('inf')

        # MODIFICADO: As constantes c1, c2, w_max, w_min agora são atributos da instância,
        # recebidos via construtor.
        self.c1 = c1 # Constante de aceleração cognitiva (peso pessoal).
        self.c2 = c2 # Constante de aceleração social (peso global).
        self.w_max = w_max # Valor máximo para o peso de inércia 'w'.
        self.w_min = w_min # Valor mínimo para o peso de inércia 'w'.

    def avaliar(self, funcao_wrapper): # Define o método 'avaliar', que calcula a aptidão (valor da função objetivo) da posição atual da partícula.
        x = self.posicao_i[0] # Extrai a coordenada x da posição atual da partícula.
        y = self.posicao_i[1] # Extrai a coordenada y da posição atual da partícula.

        # Chama a função objetivo (através do wrapper) com as coordenadas x e y.
        # O 'funcao_wrapper' também conta o número de vezes que a função objetivo é avaliada.
        self.valor_atual_i = funcao_wrapper(x, y)

        if self.valor_atual_i < self.melhor_valor_i: # Compara o valor atual com o melhor valor individual já encontrado.
            self.melhor_posicao_i = np.array(self.posicao_i) # Se o valor atual for melhor, atualiza 'melhor_posicao_i'.
            self.melhor_valor_i = self.valor_atual_i # Atualiza 'melhor_valor_i' com o novo melhor valor.

    def atualizar_velocidade(self, pos_best_g, iteracao_atual, num_iteracoes): # Define o método para atualizar a velocidade da partícula.
        # 'pos_best_g' é a melhor posição encontrada por qualquer partícula em todo o enxame (melhor global).
        # 'iteracao_atual' é a iteração atual do PSO.
        # 'num_iteracoes' é o número total de iterações do PSO.

        # MODIFICADO: w_max e w_min agora são acessados como atributos da instância (self.w_max, self.w_min).
        # Eles não são mais definidos aqui dentro da função, mas passados no construtor da Enxame.
        
        # --- Contagem de operações para o cálculo de 'w' ---
        # A linha abaixo conta as operações aritméticas envolvidas no cálculo de 'w'.
        global_op_counter.add_div(1) # Adiciona 1 à contagem de divisões (para '/ num_iteracoes').
        global_op_counter.add_mult(1) # Adiciona 1 à contagem de multiplicações (para '* (self.w_max - self.w_min)').
        
        # Calcula o peso de inércia 'w'. Ele decai linearmente de 'w_max' para 'w_min' ao longo das iterações.
        # Um 'w' decrescente ajuda a promover a exploração no início e a explotação no final.
        w = self.w_max - (iteracao_atual / num_iteracoes) * (self.w_max - self.w_min)

        r1 = random.random() # Gera um número aleatório 'r1' (entre 0 e 1) para a componente cognitiva.
        r2 = random.random() # Gera um número aleatório 'r2' (entre 0 e 1) para a componente social.

        # --- Componente cognitiva (pessoal) ---
        # Esta seção calcula a parte da velocidade que atrai a partícula para sua 'melhor_posicao_i'.
        global_op_counter.add_mult(2) # Conta 2 multiplicações (c1 * r1, e o resultado * vetor).
        vel_cognitiva = self.c1 * r1 * (self.melhor_posicao_i - self.posicao_i)

        # --- Componente social ---
        # Esta seção calcula a parte da velocidade que atrai a partícula para a 'pos_best_g' (melhor posição global).
        vel_social = self.c2 * r2 * (np.array(pos_best_g) - self.posicao_i) # Converte 'pos_best_g' para array NumPy para operação de vetor.
        # --- Atualização da velocidade final ---
        # Combina a velocidade anterior (influenciada por 'w'), a componente cognitiva e a componente social.
        global_op_counter.add_mult(2) # Conta 2 multiplicações (w * velocidade_i, para cada elemento do vetor).
        self.velocidade_i = w * self.velocidade_i + vel_cognitiva + vel_social

    def atualizar_posicao(self, limites): # Define o método para atualizar a posição da partícula com base na nova velocidade.
        # Adiciona a velocidade atual à posição atual para obter a nova posição.
        self.posicao_i = self.posicao_i + self.velocidade_i

        # Garante que a coordenada x da partícula esteja dentro dos limites definidos.
        # np.clip(valor, min, max) limita o 'valor' entre 'min' e 'max'.
        self.posicao_i[0] = np.clip(self.posicao_i[0], limites[0][0], limites[0][1])
        # Garante que a coordenada y da partícula esteja dentro dos limites definidos.
        self.posicao_i[1] = np.clip(self.posicao_i[1], limites[1][0], limites[1][1])
        
        # Verifica se a partícula atingiu ou ultrapassou os limites na coordenada x.
        if self.posicao_i[0] == limites[0][0] or self.posicao_i[0] == limites[0][1]:
            self.velocidade_i[0] = 0 # Se sim, zera a componente x da velocidade para "parar" a partícula no limite.
        # Verifica se a partícula atingiu ou ultrapassou os limites nos limites na coordenada y.
        if self.posicao_i[1] == limites[1][0] or self.posicao_i[1] == limites[1][1]:
            self.velocidade_i[1] = 0 # Se sim, zera a componente y da velocidade para "parar" a partícula no limite.