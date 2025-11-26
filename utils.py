# utils.py

class OperationCounter: # Define a classe 'OperationCounter', que é usada para rastrear o número de operações aritméticas (multiplicações, divisões, adições/subtrações) que ocorrem durante a execução do algoritmo.
    def __init__(self): # Define o método construtor da classe 'OperationCounter'. Ele é chamado quando uma nova instância da classe é criada.
        self.multiplications = 0 # Inicializa um contador para o número de multiplicações, começando em zero.
        self.divisions = 0 # Inicializa um contador para o número de divisões, começando em zero.
        self.additions = 0 # Inicializa um contador para o número de adições (e subtrações, pois são computacionalmente similares), começando em zero.

    def reset(self): # Define o método 'reset', que zera todos os contadores de operações. Isso é útil para iniciar uma nova contagem a cada execução do algoritmo.
        self.multiplications = 0 # Zera o contador de multiplicações.
        self.divisions = 0 # Zera o contador de divisões.
        self.additions = 0 # Zera o contador de adições.

    def add_mult(self, count=1): # Define o método 'add_mult', usado para adicionar um valor ao contador de multiplicações. Por padrão, adiciona 1.
        self.multiplications += count # Incrementa o contador de multiplicações pelo valor de 'count'.

    def add_div(self, count=1): # Define o método 'add_div', usado para adicionar um valor ao contador de divisões. Por padrão, adiciona 1.
        self.divisions += count # Incrementa o contador de divisões pelo valor de 'count'.

global_op_counter = OperationCounter() # Cria uma instância global da classe 'OperationCounter'. Este objeto será compartilhado e usado por todas as partes do código que precisam contar operações (como o PSO e o AG).

class FuncaoObjetivoWrapper: # Define a classe 'FuncaoObjetivoWrapper'. Esta classe atua como um "invólucro" para a sua função objetivo real (por exemplo, `minha_funcao_w29_w1`). Ela permite contar quantas vezes a função objetivo é chamada e, opcionalmente, integrar a contagem de operações.
    def __init__(self, original_func, op_counter): # Define o método construtor da classe 'FuncaoObjetivoWrapper'. Recebe a função objetivo original e uma instância do contador de operações.
        self.original_func = original_func # Armazena uma referência à função objetivo original que será envolvida (por exemplo, `minha_funcao_w29_w1`).
        self.evaluations = 0 # Inicializa um contador para o número de vezes que a função objetivo foi avaliada (chamada), começando em zero.
        self.op_counter = op_counter # Armazena uma referência à instância de 'OperationCounter' (global_op_counter), embora o wrapper em si não a use para a contagem interna da função objetivo, mas a repassa para ela.

    def __call__(self, x, y): # Permite que uma instância de 'FuncaoObjetivoWrapper' seja chamada como se fosse uma função (ex: `wrapper_instance(x, y)`).
        self.evaluations += 1 # Incrementa o contador de avaliações cada vez que a função é chamada.
        
        return self.original_func(x, y) # Chama a função objetivo original com os argumentos 'x' e 'y' e retorna o seu resultado. As operações internas da 'original_func' (se ela mesma for instrumentada) ou das chamadas no AG/PSO é que incrementarão o 'op_counter'.