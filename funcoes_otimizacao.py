import numpy as np
from utils import global_op_counter 

def calculate_z(x_original, y_original):
    """
    Calcula a componente 'z' (Schwefel).
    """
    # -x_original * np.sin(np.sqrt(np.abs(x_original)))
    global_op_counter.add_mult(1) # multiplicação externa
    global_op_counter.add_mult(1) # sqrt
    global_op_counter.add_mult(1) # sin

    term1 = -x_original * np.sin(np.sqrt(np.abs(x_original)))

    # -y_original * np.sin(np.sqrt(np.abs(y_original)))
    global_op_counter.add_mult(1) # multiplicação externa
    global_op_counter.add_mult(1) # sqrt
    global_op_counter.add_mult(1) # sin

    term2 = -y_original * np.sin(np.sqrt(np.abs(y_original)))

    return term1 + term2

def calculate_r(x_scaled, y_scaled):
    """
    Calcula a componente 'r' (Rosenbrock com o fator 100).
    """
    # x_scaled**2
    global_op_counter.add_mult(1)
    term_x_squared = x_scaled**2

    # (y_scaled - term_x_squared)**2
    global_op_counter.add_mult(1)
    inner_term_squared = (y_scaled - term_x_squared)**2

    # 100 * inner_term_squared
    global_op_counter.add_mult(1)
    part1 = 100 * inner_term_squared

    # (1 - x_scaled)**2
    global_op_counter.add_mult(1)
    part2 = (1 - x_scaled)**2

    return part1 + part2

def minha_funcao_w29_w1(x, y):
    """
    Calcula a função W29 + W1 com contagem de operações.
    """
    # 1. Z usa coordenadas originais
    z_val = calculate_z(x, y) 

    # 2. R e outros usam coordenadas escaladas
    # x / 250.0 e y / 250.0
    global_op_counter.add_div(1)
    global_op_counter.add_div(1)
    x_s = x / 250.0
    y_s = y / 250.0

    # Calcula R (Rosenbrock padrão com 100)
    r_val = calculate_r(x_s, y_s)

    # --- CÁLCULOS ESPECÍFICOS PARA W29 + W1 ---
    
    # Precisamos de r1 (sem o 100) para calcular rd
    # r1 = (y - x^2)^2 + (1 - x)^2
    # Recalculando r1 para contar as operações corretamente conforme o fluxo matemático
    
    global_op_counter.add_mult(1) # x^2
    term_x_sq = x_s**2
    
    global_op_counter.add_mult(1) # (y - x^2)^2
    term_1_sq = (y_s - term_x_sq)**2
    
    global_op_counter.add_mult(1) # (1 - x)^2
    term_2_sq = (1 - x_s)**2
    
    r1 = term_1_sq + term_2_sq
    
    # rd = 1 + r1 (soma, não conta)
    rd = 1 + r1
    
    # w14 = z * exp(sin(r1))
    global_op_counter.add_mult(1) # sin(r1)
    global_op_counter.add_mult(1) # exp(...)
    global_op_counter.add_mult(1) # z * ...
    w14 = z_val * np.exp(np.sin(r1))
    
    # w23 = z / rd
    global_op_counter.add_div(1) # divisão
    w23 = z_val / rd
    
    # w1 = r + z (soma, não conta)
    w1 = r_val + z_val
    
    # w29 = w14 + w23 (soma, não conta)
    w29 = w14 + w23
    
    # Resultado final: w29 + w1
    return w29 + w1

# --- BLOCO DE TESTE ---
if __name__ == '__main__':
    from utils import global_op_counter
    
    # Teste com um ponto único
    x_single = 10.0
    y_single = -10.0
    
    global_op_counter.reset()
    resultado = minha_funcao_w29_w1(x_single, y_single)
    
    print(f"\n--- Avaliação Única em ({x_single}, {y_single}) ---")
    print(f"Resultado: {resultado:.4f}")
    print(f"Multiplicações: {global_op_counter.multiplications}")
    print(f"Divisões: {global_op_counter.divisions}")