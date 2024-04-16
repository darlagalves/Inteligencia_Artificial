# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:53:53 2024

@author: darla
"""

import numpy as np

pesoV = np.random.uniform(-0.5, 0.5, size=(3, 2))
bias_1 = np.random.uniform(-0.5, 0.5, size=(1, 2))
alfa = 0.01
entrada = np.array([[1, 0.5, -1],
                   [0, 0.5, 1],
                   [1, -0.5, -1]])

target = np.array([[1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, 1]])

#zin = (entrada_1 * pesoV) + bias_1
def treinar (entrada, pesoV , bias_1):
    for j in range(3):
        #feedforward
        pesoW = np.random.uniform(-0.5, 0.5, size=(2, 3))
        bias_2 = np.random.uniform(-0.5, 0.5, size=(1, 3 ))
        
        zin = (entrada[j] @ pesoV) + bias_1
        tangente_hiperbolica = np.tanh(zin)
        print("Cálculo com a Entrada: ", j)
        print ("\n\n")
        print("zin: \n", zin)
        print ("\n")
        print("zj tangente: \n",tangente_hiperbolica)
        print ("\n")
        
        #yin = z * peso + bias
        yin = (zin @ pesoW) + bias_2
        tangente_hiperbolica_y = np.tanh(yin)
        print("yin: \n",yin)
        print ("\n")
        print("yk tangente: \n", tangente_hiperbolica_y)
        print ("\n")
        
        errototal = 0.5 * np.sum((target[j] - tangente_hiperbolica_y[0]) ** 2)  # Calcula o erro total
        print("Erro total:", errototal)
        print ("\n")

        
        #back propagation
        
        #deltinha = (tk - yk) *  (1 + yk) * (1 - yk)
        deltinha = (target[j] - tangente_hiperbolica_y) * ( 1 + tangente_hiperbolica_y) * (1 - tangente_hiperbolica_y)
        print("deltinha: \n",deltinha)
        print ("\n")
        
        # Transforma deltinha 
        deltinha_transformada = deltinha.reshape((-1, 1))

        
        #delta_w = alfa * deltinha * z (tangente_hiperbolica)
        delta_w = alfa * (deltinha_transformada @ tangente_hiperbolica)
        print("delta w: \n",delta_w)
        print ("\n")
        
        #delta_w0 = alfa * deltinha
        delta_w0 = alfa * deltinha
        print("delta w0: ", delta_w0)
        print ("\n")
                
        deltinha_in = deltinha_transformada.T @ pesoW.T
        print("deltinha in : ", deltinha_in)
        print ("\n")
        
        deltinha_j = deltinha_in *  ( 1 + tangente_hiperbolica) * (1 - tangente_hiperbolica)
        print("deltinha j : ", deltinha_j)
        print ("\n")
        
        entrada_aux = np.reshape(entrada[j], (1, -1))


        
        delta_v = alfa * (deltinha_j.T @ entrada_aux )
        print("delta v : ", delta_v)
        print ("\n")
        
        delta_v0 = alfa * deltinha_j
        
        w_novo = pesoW + delta_w.T
        
        w0_novo = bias_2 + delta_w0.T
        
        v_novo = pesoV + delta_v.T
        
        v0_novo = bias_1 + delta_v0
        
        #atualizando valores de peso e bias
        pesoV = v_novo
        bias_1 = v0_novo
        pesoW = w_novo
        bias_2 = w0_novo
    return pesoV,bias_1,pesoW,bias_2

    
pesoV_final, bias_1_final, pesoW_final, bias_2_final = treinar(entrada, pesoV , bias_1)

def teste(inputs, pesoV, pesoW, bias_1, bias_2):
    # Teste da MLP
    outputs = []
    for input_data in inputs:
        zin = (input_data @ pesoV) + bias_1
        tangente_hiperbolica = np.tanh(zin)
        yin = (zin @ pesoW) + bias_2
        tangente_hiperbolica_y = np.tanh(yin)
        outputs.append(tangente_hiperbolica_y)

    return outputs

# Dados de teste
test_inputs = [[1, 0.5, -1], [0, 0.5, 1], [1, -0.5, -1]]

# Teste da MLP
test_outputs = teste(test_inputs, pesoV_final, pesoW_final, bias_1_final, bias_2_final)

print("Outputs após o teste:")
for output in test_outputs:
    print(output)
