# Genetico.py

# Importa as bibliotecas necessárias para o Algoritmo Genético
import numpy as np # NumPy para operações numéricas e manipulação de arrays
import random # Para geração de números aleatórios
import matplotlib.pyplot as plt # Matplotlib para criação de gráficos
from mpl_toolkits.mplot3d import Axes3D 
import os # Para interagir com o sistema operacional (criação de pastas, manipulação de arquivos)
from datetime import datetime # Para trabalhar com datas e horas (gerar timestamps para nomes de arquivos)

# Importa as funções personalizadas do projeto
from funcoes_otimizacao import minha_funcao_w29_w1 # A função objetivo W29 + W1 a ser minimizada
from Grafico import GraficoAG # Importa a função para plotar o gráfico do AG (visualização 3D)
from utils import global_op_counter, FuncaoObjetivoWrapper # Contadores globais e o wrapper para a função objetivo

# --- Funções Auxiliares do Algoritmo Genético ---

def inicializar_populacao(tamanho_populacao, limites):
    """
    Inicializa a população inicial do algoritmo genético.
    Cada indivíduo é um par (x, y) gerado aleatoriamente dentro dos limites.
    
    Parâmetros:
        tamanho_populacao (int): O número de indivíduos na população.
        limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca.
        
    Retorna:
        list: Uma lista de arrays NumPy, onde cada array representa um indivíduo (x, y).
    """
    populacao = []
    for _ in range(tamanho_populacao):
        individuo = np.array([
            random.uniform(limites[0], limites[1]),
            random.uniform(limites[0], limites[1]),
        ])
        populacao.append(individuo)
    return populacao

def avaliar_populacao(populacao, funcao_wrapper):
    """
    Avalia a aptidão (valor da função objetivo) de cada indivíduo na população.
    
    Parâmetros:
        populacao (list): A lista de indivíduos da população.
        funcao_wrapper (FuncaoObjetivoWrapper): Um wrapper da função objetivo que conta as avaliações.
        
    Retorna:
        list: Uma lista de aptidões, onde cada aptidão corresponde a um indivíduo.
    """
    aptidoes = []
    for individuo in populacao:
        x, y = individuo
        aptidao = funcao_wrapper(x, y)
        aptidoes.append(aptidao)
    return aptidoes

def selecao_roleta(populacao, aptidoes):
    """
    Seleciona um indivíduo da população usando o método de seleção por roleta.
    Retorna o indivíduo selecionado.
    Como é minimização, transformamos a aptidão para que valores menores tenham maior "fatia" na roleta.
    
    Parâmetros:
        populacao (list): A lista de indivíduos da população.
        aptidoes (list): A lista de aptidões correspondentes aos indivíduos.
        
    Retorna:
        np.array or None: O indivíduo selecionado ou None se a lista de aptidões estiver vazia.
    """
    if not aptidoes:
        return None

    max_aptidao = max(aptidoes)
    # Transforma as aptidões para que valores menores resultem em scores maiores (para minimização).
    fit_scores = [max_aptidao - apt + 1e-6 for apt in aptidoes] 
    
    # Calcula a soma total dos fit_scores.
    # Este é o 'tamanho' total da roleta que será usada para a seleção.
    total_fit = sum(fit_scores)

    # Verifica uma condição de borda: se a soma total dos scores for zero.
    # Isso pode ocorrer se todas as aptidões forem iguais e a transformação 'max_aptidao - apt + 1e-6'
    # resultar em valores muito próximos de zero, ou se houver algum problema inesperado.
    # Nesses casos, a seleção proporcional não pode ser feita.
    if total_fit == 0:
        # Se a soma é zero, todos os indivíduos têm a mesma "aptidão" (ou problema).
        # Para evitar divisão por zero ou travamento, um indivíduo é escolhido aleatoriamente.
        return random.choice(populacao)

    # Calcula a probabilidade de seleção para cada indivíduo.
    # Cada probabilidade é a 'fatia' do indivíduo na roleta, proporcional ao seu fit_score.
    # A soma de todas essas probabilidades será 1.
    probabilidades = [score / total_fit for score in fit_scores]

    # Gera um número aleatório entre 0.0 (inclusive) e 1.0 (exclusive).
    # Este número simula o 'giro da roleta' e a posição onde a 'seta' parou.
    r = random.random()

    # Inicializa uma variável para somar as probabilidades cumulativamente.
    # Conforme iteramos pelos indivíduos, 'acumulado' representará o limite superior
    # da fatia da roleta que inclui todos os indivíduos até o atual.
    acumulado = 0

    # Itera sobre a lista de probabilidades, juntamente com seus índices.
    # O 'enumerate' nos dá o índice (i) e o valor (prob) de cada item.
    for i, prob in enumerate(probabilidades):
        # Adiciona a probabilidade do indivíduo atual à soma acumulada.
        acumulado += prob
        # Verifica se o número aleatório 'r' caiu na 'fatia' deste indivíduo.
        # Se 'r' for menor ou igual ao 'acumulado', significa que o indivíduo 'i' foi selecionado.
        if r <= acumulado:
            # Retorna o indivíduo da população correspondente ao índice selecionado.
            return populacao[i]

    # Este é um fallback (plano de segurança).
    # Na teoria, o loop deve sempre encontrar um indivíduo antes de terminar,
    # pois 'r' está entre 0 e 1, e a soma das probabilidades é 1.
    # No entanto, em casos raríssimos (e.g., problemas de ponto flutuante),
    # pode ser que 'r' seja ligeiramente maior que o último 'acumulado'
    # mesmo que 'acumulado' seja 1.0. Para garantir que a função sempre retorne algo,
    # o último indivíduo da população é retornado.
    return populacao[-1]


def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)):
    """
    Realiza o cruzamento BLX-alpha (Blend Crossover Alpha) entre os DOIS pais fornecidos.
    Gera DOIS novos filhos com base nos genes dos pais, com uma chance de ocorrer o cruzamento.
    
    Parâmetros:
        pais (list): Uma lista contendo dois indivíduos (pais) para cruzamento.
                     Cada pai é esperado ser um np.array, e ter a mesma dimensão (número de genes).
        taxa_cruzamento (float): A probabilidade (entre 0 e 1) de ocorrer o cruzamento.
                                 Se um número aleatório for menor que esta taxa, o cruzamento ocorre.
        alpha (float): Parâmetro alfa do BLX-alpha. Define a "largura" da faixa de mistura
                       dos genes dos pais. Geralmente entre 0 e 1, mas pode ser maior.
                       Um valor de 0.5 (padrão) é comum.
        limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca
                         para cada gene. Isso garante que os filhos gerados permaneçam
                         dentro dos limites válidos do problema.
        
    Retorna:
        list: Uma lista contendo os dois filhos gerados pelo cruzamento (np.arrays),
              ou cópias dos pais originais se o cruzamento não ocorreu.
    """
    # Desempacota os dois pais da lista 'pais'.
    # pai1 e pai2 são esperados ser arrays NumPy, representando os genes (x, y) dos indivíduos.
    pai1, pai2 = pais[0], pais[1]

    # Decide se o cruzamento realmente acontecerá com base na 'taxa_cruzamento'.
    # Gera um número aleatório entre 0 e 1. Se esse número for menor que a taxa_cruzamento, o cruzamento é ativado.
    if random.random() < taxa_cruzamento:
        # Se o cruzamento ocorre:

        # Cria dois novos arrays NumPy, 'filho1' e 'filho2', preenchidos com zeros.
        # Eles terão a mesma forma (dimensões) que os pais.
        # Estes arrays serão preenchidos com os genes dos novos filhos.
        filho1 = np.zeros_like(pai1)
        filho2 = np.zeros_like(pai2)

        # Itera sobre cada gene (ou dimensão) dos pais.
        # Para um problema 2D (x, y), len(pai1) será 2.
        for i in range(len(pai1)):
            # Obtém o valor do gene atual para cada pai.
            gene1_pai = pai1[i]
            gene2_pai = pai2[i]

            # Calcula a distância absoluta entre os genes dos dois pais para a dimensão atual.
            # 'I' de "interval" ou "difference".
            I = abs(gene1_pai - gene2_pai)
            
            # Conta uma operação de multiplicação para fins de benchmarking/estatísticas.
            global_op_counter.add_mult(1) 
            # Calcula 'd', que define a extensão do intervalo de mistura além dos genes dos pais.
            # Este é o "blend" do BLX (Blend Crossover).
            # Por exemplo, se alpha=0.5 e I=10, d será 5.
            d = alpha * I

            # Calcula os limites inferiores e superiores do intervalo de onde os genes dos filhos
            # serão selecionados. Este intervalo se estende para fora dos genes dos pais
            # por uma margem 'd' em ambas as direções.
            # Ex: se gene1_pai=10, gene2_pai=20, I=10, d=5 (com alpha=0.5).
            # min(10,20)-5 = 10-5 = 5
            # max(10,20)+5 = 20+5 = 25
            # O intervalo inicial de seleção será [5, 25].
            lower_bound = min(gene1_pai, gene2_pai) - d
            upper_bound = max(gene1_pai, gene2_pai) + d

            # Garante que os limites calculados para o gene dos filhos não ultrapassem
            # os limites globais definidos para o espaço de busca do problema.
            # Isso é crucial para manter os filhos dentro de uma região válida.
            lower_bound = max(lower_bound, limites[0]) # Garante que lower_bound não seja menor que limites[0]
            upper_bound = min(upper_bound, limites[1]) # Garante que upper_bound não seja maior que limites[1]

            # Seleciona aleatoriamente o valor do gene para o filho1 dentro do intervalo ajustado.
            filho1[i] = random.uniform(lower_bound, upper_bound)
            # Seleciona aleatoriamente o valor do gene para o filho2 dentro do mesmo intervalo ajustado.
            filho2[i] = random.uniform(lower_bound, upper_bound)
            
        # Retorna uma lista contendo os dois filhos recém-criados.
        return [filho1, filho2]
    else:
        # Se o cruzamento NÃO ocorre (random.random() >= taxa_cruzamento):
        # Retorna cópias dos pais originais. É importante usar .copy() para evitar
        # que as referências sejam as mesmas e modificações futuras nos filhos
        # afetem os pais originais.
        return [pai1.copy(), pai2.copy()]

def mutacao(populacao, taxa_mutacao, limites):
    """
    Aplicação de mutação a indivíduos da população.
    Esta função implementa a MUTACÃO UNIFORME para indivíduos com representação de ponto flutuante.
    A mutação uniforme substitui um gene aleatório do indivíduo por um novo valor
    também aleatório, mas dentro dos limites definidos do espaço de busca.
    
    Parâmetros:
        populacao (list): A lista de objetos ou arrays NumPy, onde cada elemento representa um indivíduo.
                          Esperamos que cada indivíduo seja um vetor (e.g., [x, y]).
        taxa_mutacao (float): A probabilidade (entre 0 e 1) de UM INDIVÍDUO sofrer mutação.
                              Se um indivíduo for selecionado para mutação, APENAS UM de seus genes
                              será alterado (selecionado aleatoriamente).
        limites (tuple): Uma tupla (min, max) definindo os limites inferior e superior
                         do espaço de busca. Os novos valores de gene gerados pela mutação
                         estarão dentro desses limites.
        
    Retorna:
        list: A população após a aplicação das mutações. Os indivíduos na lista original
              podem ter sido modificados "in-place" se sofreram mutação.
    """
    # Itera sobre cada 'individuo' na 'populacao' fornecida.
    # A mutação é aplicada individualmente, com base em uma probabilidade.
    for individuo in populacao:
        # Decide se este 'individuo' em particular sofrerá mutação.
        # Gera um número aleatório entre 0.0 e 1.0.
        # Se este número for menor que 'taxa_mutacao', o indivíduo é escolhido para mutar.
        if random.random() < taxa_mutacao:
            # Se o indivíduo vai mutar:

            # Escolhe aleatoriamente QUAL gene (dimensão, e.g., 'x' ou 'y') do indivíduo será mutado.
            # 'len(individuo) - 1' garante que o índice esteja dentro dos limites do vetor do indivíduo.
            # Por exemplo, se individuo for [x, y], len(individuo) é 2, então random.randint(0, 1)
            # pode retornar 0 (para x) ou 1 (para y).
            indice_mutacao = random.randint(0, len(individuo) - 1) 
            
            # Aplica a MUTACÃO UNIFORME:
            # O gene selecionado ('individuo[indice_mutacao]') é substituído por um NOVO valor
            # gerado aleatoriamente dentro dos 'limites' definidos do espaço de busca.
            # Esta é a característica principal da mutação uniforme: o novo valor não depende
            # do valor original do gene, apenas dos limites.
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1])
            
    # Retorna a população.
    return populacao
# --- ALGORITMO GENÉTICO PRINCIPAL ---
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50, tolerancia=1e-6):
    """
    Implementa o algoritmo genético principal para otimização.
    Este é o orquestrador do AG, controlando o ciclo de gerações, seleção,
    cruzamento, mutação e o monitoramento da convergência.
    
    Parâmetros:
        tamanho_populacao (int): O número de indivíduos que a população terá em cada geração.
        limites (tuple): Uma tupla (min, max) definindo os limites inferior e superior
                         para os valores dos genes (coordenadas x, y) dos indivíduos.
        num_geracoes (int): O número máximo de gerações que o algoritmo irá executar.
                            É um critério de parada.
        taxa_cruzamento (float): A probabilidade (entre 0 e 1) de um par de pais
                                 realizar o cruzamento para gerar filhos.
        taxa_mutacao (float): A probabilidade (entre 0 e 1) de um indivíduo
                              sofrer mutação em um de seus genes.
        geracoes_sem_melhora_limite (int): Limite de gerações consecutivas sem uma
                                           melhora significativa na melhor aptidão global.
                                           Se atingido, o algoritmo para (critério de parada por convergência).
        tolerancia (float): Um pequeno valor para definir o que é considerado uma "melhora significativa".
                            Se a diferença entre a melhor aptidão atual e a anterior for menor que a tolerância,
                            não é considerada uma melhora real.

    Retorna:
        dict: Um dicionário contendo os resultados e estatísticas finais do algoritmo,
              incluindo os históricos de convergência para análise posterior.
    """
    # Reinicia os contadores globais de operações para o início de uma nova execução do AG.
    # Isso é útil para comparar o custo computacional entre diferentes execuções ou algoritmos.
    global_op_counter.reset()
    # Cria uma 'wrapper' para a função objetivo (minha_funcao_w29_w1).
    # Esta wrapper permite contar o número de vezes que a função objetivo é avaliada
    # e passa essas avaliações para o contador global de operações, caso a função w29 + w1
    # tenha operações internas para contar.
    minha_funcao_w29_w1_wrapper_ag = FuncaoObjetivoWrapper(minha_funcao_w29_w1, global_op_counter)

    # --- INICIALIZAÇÃO DO ALGORITMO ---
    # Cria a população inicial de indivíduos de forma aleatória, respeitando os limites.
    populacao = inicializar_populacao(tamanho_populacao, limites)
    # Inicializa a melhor solução encontrada até agora como Nula.
    melhor_solucao = None
    # Inicializa a melhor aptidão (valor da função objetivo) como infinito positivo.
    # Isso é feito para problemas de MINIMIZAÇÃO, garantindo que qualquer aptidão real
    # encontrada na primeira geração será considerada uma melhoria.
    melhor_aptidao = float('inf')
    # Registra a geração em que a melhor solução global foi encontrada pela primeira vez.
    melhor_geracao = -1

    # Variáveis para armazenar o número de avaliações e operações quando a melhor solução global foi encontrada.
    avaliacoes_ag_melhor_solucao = 0
    operacoes_ag_melhor_solucao_mult = 0
    operacoes_ag_melhor_solucao_div = 0

    # Contador para o critério de parada por falta de melhora.
    geracoes_sem_melhora = 0
    # Armazena o valor da melhor aptidão da geração anterior para comparar com a atual.
    ultima_melhor_aptidao_global = float('inf') # Inicializado como infinito para a primeira comparação.

    # --- LISTAS PARA ARMAZENAR O HISTÓRICO DE CONVERGÊNCIA ---
    # Armazena a melhor aptidão encontrada em CADA GERAÇÃO.
    historico_melhor_geracao = []
    # Armazena a melhor aptidão GLOBAL (a melhor encontrada até o momento) após cada geração.
    historico_melhor_global = []

    # --- CONFIGURAÇÃO DO GRÁFICO (para visualização em 3D) ---
    # Cria uma figura para o gráfico.
    fig = plt.figure(figsize=(11, 8))
    # Adiciona um subplot 3D à figura.
    ax = fig.add_subplot(111, projection='3d')
    # Ajusta o espaçamento da figura para acomodar o texto ao lado do gráfico.
    plt.subplots_adjust(right=0.7)

    # Dicionário para passar parâmetros do AG para a função de plotagem.
    # Contém informações estáticas e dinâmicas que serão atualizadas a cada geração.
    ag_params_for_plot = {
        "tamanho_populacao": tamanho_populacao,
        "taxa_mutacao": taxa_mutacao,
        "taxa_crossover": taxa_cruzamento,
        "selecao_tipo": "Roleta", # Define o tipo de seleção usado.
        "iteracoes_totais": num_geracoes,
        "avaliacoes_funcao": 0, # Será atualizado.
        "multiplicacoes_total": 0, # Será atualizado.
        "divisoes_total": 0, # Será atualizado.
        "avaliacoes_minimo_global": 0, # Será atualizado quando o mínimo global for encontrado/melhorado.
        "multiplicacoes_minimo_global": 0, # Será atualizado.
        "divisoes_minimo_global": 0, # Será atualizado.
        "geracoes_sem_melhora": geracoes_sem_melhora, # Será atualizado.
        "limite_geracoes_sem_melhora": geracoes_sem_melhora_limite
    }

    # --- LOOP PRINCIPAL DO ALGORITMO GENÉTICO (POR GERAÇÕES) ---
    for i in range(num_geracoes): 
        # Avalia a aptidão (valor da função objetivo) de cada indivíduo na população atual.
        # Retorna uma lista de aptidões, onde aptidoes[j] corresponde ao indivíduo j.
        aptidoes = avaliar_populacao(populacao, minha_funcao_w29_w1_wrapper_ag) 

        # --- Tratamento de casos onde a avaliação da população falha ou retorna valores inválidos ---
        # Verifica se a lista de aptidões está vazia ou se o menor valor é infinito (problema na avaliação).
        if not aptidoes or np.isinf(min(aptidoes)):
            # Se não houver aptidões válidas, adiciona o último melhor global (ou infinito)
            # ao histórico para manter a continuidade dos gráficos.
            if historico_melhor_global:
                historico_melhor_geracao.append(historico_melhor_global[-1])
                historico_melhor_global.append(historico_melhor_global[-1])
            else:
                # Se ainda não há histórico, preenche com infinito.
                historico_melhor_geracao.append(float('inf'))
                historico_melhor_global.append(float('inf'))
            
            # Atualiza e exibe o gráfico mesmo com problemas na avaliação, para não travar o loop.
            GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)
            # Pula para a próxima iteração do loop, sem executar o restante da lógica de AG para esta geração.
            continue

        # Encontra a melhor aptidão (menor valor da função objetivo) na geração atual.
        melhor_aptidao_geracao = min(aptidoes)
        # Adiciona a melhor aptidão da geração ao histórico específico de geração.
        historico_melhor_geracao.append(melhor_aptidao_geracao)

        # --- ATUALIZAÇÃO DA MELHOR SOLUÇÃO GLOBAL ---
        # Compara a melhor aptidão da geração atual com a melhor aptidão global encontrada até agora.
        if melhor_aptidao_geracao < melhor_aptidao:
            # Se a melhor aptidão da geração atual é "realmente" melhor (considerando a tolerância):
            # A diferença absoluta é usada para verificar se a melhora é significativa e não apenas ruído.
            if abs(melhor_aptidao - melhor_aptidao_geracao) > tolerancia:
                # Atualiza a melhor aptidão global.
                melhor_aptidao = melhor_aptidao_geracao
                # Atualiza a melhor solução (o indivíduo) global. Usa .copy() para garantir
                # que estamos armazenando uma cópia do array e não uma referência que poderia ser alterada.
                melhor_solucao = populacao[aptidoes.index(melhor_aptidao_geracao)].copy()
                # Registra a geração em que esta nova melhor solução global foi encontrada.
                melhor_geracao = i
                
                # Armazena o estado atual dos contadores globais, pois esta é a "melhor solução" até agora.
                avaliacoes_ag_melhor_solucao = minha_funcao_w29_w1_wrapper_ag.evaluations
                operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications
                operacoes_ag_melhor_solucao_div = global_op_counter.divisions
                
                # Reseta o contador de gerações sem melhora, pois uma melhora significativa foi encontrada.
                geracoes_sem_melhora = 0
            else:
                # Se a melhora é menor que a tolerância, ainda contamos como uma geração sem melhora significativa.
                geracoes_sem_melhora += 1
        else:
            # Se a melhor aptidão da geração atual NÃO é melhor que a melhor global,
            # incrementa o contador de gerações sem melhora.
            geracoes_sem_melhora += 1
            
        # Adiciona a melhor aptidão global (acumulativa) ao histórico global.
        # Mesmo que não tenha havido melhora nesta geração, o valor armazenado é o melhor até o momento.
        historico_melhor_global.append(melhor_aptidao)

        # --- CRITÉRIO DE PARADA POR CONVERGÊNCIA ---
        # Verifica se o número de gerações sem melhora atingiu o limite definido.
        if geracoes_sem_melhora >= geracoes_sem_melhora_limite: 
            # Imprime uma mensagem indicando o motivo da parada antecipada.
            print(f"\n[AG] Parada por convergência: Mudança no melhor valor da aptidão menor que {tolerancia} por {geracoes_sem_melhora_limite} gerações. (Geração: {i + 1})")
            # Garante que o último frame do gráfico seja atualizado e mostrado antes de fechar.
            GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)
            # Exibe o gráfico final e pausa a execução até que o usuário o feche.
            plt.show(block=True) 
            # Interrompe o loop principal do AG.
            break
            
        # --- GERAÇÃO DA PRÓXIMA POPULAÇÃO ---
        proxima_geracao = [] # Lista para armazenar os indivíduos da nova geração.
        
        # Loop para gerar a próxima população através de seleção e cruzamento.
        # Itera tamanho_populacao / 2 vezes para gerar pares de filhos.
        for _ in range(tamanho_populacao // 2): 
            # Seleciona dois pais da população atual usando o método da roleta.
            pai1 = selecao_roleta(populacao, aptidoes)
            pai2 = selecao_roleta(populacao, aptidoes)

            # Verifica se os pais foram selecionados com sucesso (evita None se houver problemas na seleção).
            if pai1 is None or pai2 is None:
                continue # Se um pai não foi selecionado, pula para a próxima iteração.

            # Realiza o cruzamento BLX-alpha entre os dois pais.
            # Retorna uma lista de dois filhos (ou cópias dos pais se não houver cruzamento).
            filhos_gerados = cruzamento_blx_alpha([pai1, pai2], taxa_cruzamento, alpha=0.5, limites=limites)
            # Adiciona os filhos gerados à lista da próxima geração.
            proxima_geracao.extend(filhos_gerados)

        # Trata o caso em que o tamanho da população é ímpar, ou se alguns cruzamentos falharam.
        # Adiciona um pai extra para completar a população se necessário.
        if len(proxima_geracao) < tamanho_populacao:
            extra_pai = selecao_roleta(populacao, aptidoes)
            if extra_pai is not None:
                proxima_geracao.append(extra_pai.copy()) # Adiciona uma cópia para evitar referência direta.

        # Aplica mutação aos indivíduos da próxima geração.
        # A mutação é feita "in-place" nos indivíduos da lista 'proxima_geracao'.
        filhos_apos_mutacao = mutacao(proxima_geracao, taxa_mutacao, limites) 

        # A nova população é composta pelos primeiros 'tamanho_populacao' indivíduos resultantes
        # do cruzamento e mutação. Isso garante que o tamanho da população seja mantido constante.
        populacao = filhos_apos_mutacao[:tamanho_populacao] 

        # --- ATUALIZAÇÃO DOS PARÂMETROS PARA PLOTAGEM E CONTADORES ---
        # Atualiza os contadores totais de avaliações de função e operações.
        ag_params_for_plot["avaliacoes_funcao"] = minha_funcao_w29_w1_wrapper_ag.evaluations
        ag_params_for_plot["multiplicacoes_total"] = global_op_counter.multiplications
        ag_params_for_plot["divisoes_total"] = global_op_counter.divisions
        # Atualiza o contador de gerações sem melhora.
        ag_params_for_plot["geracoes_sem_melhora"] = geracoes_sem_melhora
        
        # Atualiza e exibe o gráfico da população atual e da melhor solução.
        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)

    # --- CRITÉRIO DE PARADA POR NÚMERO MÁXIMO DE GERAÇÕES ---
    # Se o loop terminou porque atingiu o número máximo de gerações (e não por convergência antecipada).
    if i + 1 == num_geracoes:
        print(f"\n[AG] Loop de gerações concluído. Total de gerações: {i + 1}")
        # Garante que o gráfico final seja exibido e aguarda o fechamento do usuário.
        plt.show(block=True) 
    
    # --- Prepara os resultados para retorno ---
    # Coleta todas as informações relevantes sobre a execução do AG.
    return {
        "melhor_solucao": melhor_solucao, # O indivíduo com o melhor valor encontrado.
        "melhor_valor_global": melhor_aptidao, # O valor da função objetivo do melhor indivíduo.
        "iteracoes_executadas": i + 1, # O número total de gerações que o algoritmo executou.
        "avaliacoes_funcao_total": minha_funcao_w29_w1_wrapper_ag.evaluations, # Total de vezes que a função objetivo foi chamada.
        "multiplicacoes_total": global_op_counter.multiplications, # Total de operações de multiplicação.
        "divisoes_total": global_op_counter.divisions, # Total de operações de divisão.
        "avaliacoes_minimo_global": avaliacoes_ag_melhor_solucao, # Av. de função até o encontro do mínimo global.
        "multiplicacoes_minimo_global": operacoes_ag_melhor_solucao_mult, # Mult. até o encontro do mínimo global.
        "divisoes_minimo_global": operacoes_ag_melhor_solucao_div, # Div. até o encontro do mínimo global.
        "historico_melhor_geracao": historico_melhor_geracao, # Histórico da melhor aptidão em cada geração.
        "historico_melhor_global": historico_melhor_global, # Histórico da melhor aptidão global ao longo das gerações.
    }

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    # Define os parâmetros de teste para uma execução direta.
    limites = (-500, 500)
    tamanho_populacao = 35
    num_geracoes = 200
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 20
    tolerancia = 1e-6

    # Chama a função do algoritmo genético com os parâmetros de teste.
    results = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite,
        tolerancia=tolerancia
    )
    # Impressão dos resultados do teste direto (usando o dicionário retornado)
    print("\n--- Resultados Finais do Teste Direto do AG ---")
    print(f"Melhor solução encontrada: {results['melhor_solucao']}")
    print(f"Melhor valor global: {results['melhor_valor_global']:.4f}")
    print(f"Gerações executadas: {results['iteracoes_executadas']}")
    print(f"Avaliações da função (total): {results['avaliacoes_funcao_total']}")
    print(f"Multiplicações (total): {results['multiplicacoes_total']}")
    print(f"Divisões (total): {results['divisoes_total']}")
    print(f"Avaliações (no melhor global): {results['avaliacoes_minimo_global']}")
    print(f"Multiplicações (no melhor global): {results['multiplicacoes_minimo_global']}")
    print(f"Divisões (no melhor global): {results['divisoes_minimo_global']}")
    print(f"Tamanho do histórico_melhor_geracao: {len(results['historico_melhor_geracao'])}")
    print(f"Tamanho do historico_melhor_global: {len(results['historico_melhor_global'])}")