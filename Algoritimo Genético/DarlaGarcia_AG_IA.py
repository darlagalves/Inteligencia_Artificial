# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:07:31 2024

@author: darla
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math
from collections import Counter

print("\x1b[2J\x1b[1;1H") 

def gerarElementos(n=513):
    vetor = [rd.randint(0, 512) for _ in range(n)]  # Cria uma lista de n números aleatórios entre 0 e 512
    print(vetor)
    return vetor

def gerarElementosBinarios(vetor_aleatorio):
    vetor_binario = []
    for numero in vetor_aleatorio:
        binario = format(numero, '010b')  # Converte o número para uma string binária de 8 bits
        vetor_binario.append(binario)
    print(vetor_binario)
    return vetor_binario

def gerarImagem(vetor_aleatorio):
    # Criando vetor vazio para armazenar as imagens
    vetor_imagem = np.empty_like(vetor_aleatorio)

    # Calculando a imagem para cada elemento do vetor aleatório
    for i in range(len(vetor_aleatorio)):
        # Função f(x)
        valor_imagem = -abs(vetor_aleatorio[i] * math.sin(math.sqrt(abs(vetor_aleatorio[i]))))

        # Armazenando a imagem na posição correspondente do vetor_imagem
        vetor_imagem[i] = valor_imagem

    # Retornando o vetor com as imagens
    return vetor_imagem

def gerarProbabilidades(vetor_imagem):
    # Calculando a "probabilidade" de cada imagem
    vetor_prob = (vetor_imagem / (sum(vetor_imagem))) * 100
    # Retornando o vetor com os valores das probabilidades
    return vetor_prob

def separarMelhores(vetor_prob, vetor_binario, n=50):
    # Combine os índices com as porcentagens
    porcentagem_com_indices = list(enumerate(vetor_prob))
    
    # Ordene pelo valor da porcentagem em ordem decrescente
    porcentagem_com_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Pegue os primeiros `n` índices após a ordenação
    indices_melhores = [indice for indice, porcentagem in porcentagem_com_indices[:n]]
    
    # Selecione os elementos do vetor_binarios usando esses índices
    vetor_melhores = [vetor_binario[indice] for indice in indices_melhores]
    
    return vetor_melhores
    
def sortearCasais(quantidade_casais=25):
    # Criando vetor vazio para armazenar os "casais"
    casais_sorteados = np.empty((quantidade_casais, 2), dtype=int)

    # Criando vetor com todos os números de 0 a 49
    numeros_possiveis = np.arange(50)

    # Percorrendo a quantidade de "casais"
    for i in range(quantidade_casais):
        # Sorteando o primeiro número do "casal"
        indice_numero1 = np.random.choice(numeros_possiveis)
    
        # Removendo o número sorteado do vetor de possibilidades
        numeros_possiveis = numeros_possiveis[numeros_possiveis != indice_numero1]
    
        # Sorteando o segundo número do "casal"
        indice_numero2 = np.random.choice(numeros_possiveis)
    
        # Armazenando os números sorteados no vetor de "casais"
        casais_sorteados[i] = [indice_numero1, indice_numero2]

    # Retornando o vetor com os "casais" sorteados
    return casais_sorteados

def gerarPontoCorte():
    numero_sorteado = rd.randint(1, 9)
    return numero_sorteado

def recombinar(numero_sorteado, casais_sorteados, vetor_melhores):
    # Criando lista vazia para armazenar os filhos
    filhos = []

    # Percorrendo os casais sorteados
    for i, casal in enumerate(casais_sorteados):
        # Separando os pais
        pai1 = vetor_melhores[casal[0]]
        pai2 = vetor_melhores[casal[1]]
        
        # Realizando o cruzamento
        filho1 = pai1[:numero_sorteado] + pai2[numero_sorteado:]  # Primeira metade do pai1 e segunda metade do pai2
        filho2 = pai2[:numero_sorteado] + pai1[numero_sorteado:]  # Primeira metade do pai2 e segunda metade do pai1
        
        # Adicionando os filhos à lista de filhos
        filhos.append(filho1)
        filhos.append(filho2)
        
    return filhos  # Retorna a lista de filhos

def gerarMutacao(filhos):  
    # Define o número de filhos que serão mutados (por exemplo, 10% dos filhos)
    num_filhos_mutados = int(len(filhos) * 0.1) 

    # Sorteia aleatoriamente os índices dos filhos que serão mutados
    indices_mutados = rd.sample(range(len(filhos)), num_filhos_mutados)

    vetor_modificado = filhos.copy()  # Cria uma cópia da lista de filhos

    for indice in indices_mutados:
        # Sorteia uma posição de bit (entre 0 e 9 para strings binárias de 10 bits)
        posicao = rd.randint(0, 9)
        print(f"Bit sorteado para alteração: {posicao}")
        # Converte a string binária em uma lista de caracteres para facilitar a modificação
        lista_binario = list(vetor_modificado[indice])
        print(vetor_modificado[indice])
        # Inverte o bit na posição sorteada
        lista_binario[posicao] = '0' if lista_binario[posicao] == '1' else '1'
        # Converte a lista de volta para string binária
        binario_modificado = ''.join(lista_binario)
        vetor_modificado[indice] = binario_modificado  # Atualiza o filho na lista

    return vetor_modificado

def misturarNovosAntigos(vetor_binario_melhores, filhos):
    # Combinar os melhores binários e os filhos em um único vetor
    vetor_combinado = vetor_binario_melhores + filhos
    return vetor_combinado

def novosValoresDecimais(vetor_binarios):
    vetor_decimais = []
    for numero_binario in vetor_binarios:
        # Converter o número binário para decimal usando a função int() e a base 2
        valor_decimal = int(numero_binario, 2)
        # Adicionar o valor decimal ao vetor de decimais
        vetor_decimais.append(valor_decimal)
    print(vetor_decimais)
    return vetor_decimais

popDec = gerarElementos()
popBin = gerarElementosBinarios(popDec)
imagemFuncao = gerarImagem(popDec)
probabRolet = gerarProbabilidades(imagemFuncao)
sMelhores = separarMelhores(probabRolet, popBin)
recomb = sortearCasais()
pontoCorte = gerarPontoCorte()
s_filhos = recombinar(pontoCorte, recomb, sMelhores)
s_filhos = gerarMutacao(s_filhos)
popBin = misturarNovosAntigos(sMelhores, s_filhos)
popDec = novosValoresDecimais(popBin)
aux = 0

# Fazendo o laço de gerações
while aux < 100:
    imagemFuncao = gerarImagem(popDec)
    probabRolet = gerarProbabilidades(imagemFuncao)
    sMelhores = separarMelhores(probabRolet, popBin)
    recomb = sortearCasais()
    pontoCorte = gerarPontoCorte()
    s_filhos = recombinar(pontoCorte, recomb, sMelhores)
    s_filhos = gerarMutacao(s_filhos)
    popBin = misturarNovosAntigos(sMelhores, s_filhos)
    popDec = novosValoresDecimais(popBin)
    # Criando um objeto Counter a partir do vetor
    contador = Counter(popDec)
    # Acessando o elemento mais frequente e sua contagem
    elemento_mais_frequente, frequencia_maxima = contador.most_common(1)[0]
    print(f"O número mais frequente no vetor é {elemento_mais_frequente} e aparece {frequencia_maxima} vezes.")
    print("===== GERAÇÃO ===== ", aux)
    aux += 1

# Gerando os valores de x no intervalo de 0 a 1024
x = np.linspace(0, 1024, 1000)  # 1000 pontos para uma boa resolução

# Calculando os valores da função para cada x
y = -abs(x * np.sin(np.sqrt(abs(x))))

# Plotando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='-abs(x * sin(sqrt(abs(x))))')

# Adicionando o ponto mais frequente ao gráfico
y_elemento_mais_frequente = -abs(elemento_mais_frequente * np.sin(np.sqrt(abs(elemento_mais_frequente))))
plt.scatter([elemento_mais_frequente], [y_elemento_mais_frequente], color='red')
plt.text(elemento_mais_frequente, y_elemento_mais_frequente, f'({elemento_mais_frequente}, {y_elemento_mais_frequente:.2f})',
         fontsize=12, ha='right')

plt.title('Gráfico da função -abs(x * sin(sqrt(abs(x)))) com o elemento mais frequente')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
