import random
import streamlit as st
import numpy as np
from collections import defaultdict
import random as rd
import matplotlib.pyplot as plt
import math
import altair as alt
from collections import Counter

#pegar informações de entrada: tamanho da popu, numr de gerações, taxa de mutação, quantidade de melhores por geração
tamanho_populacao = 100
tamanho_torneio = 8
geracoes = 1000
taxa_mutacao = 10
qtd_melhores = 20
torneio_roleta = True       # True = torneio e False = roleta
pontos = False               # True = 1 ponto e False = 2 pontos
tamanho_coluna = 5
tamanho_linha = 6
tamanho_horario = (5, 6)

#criar população - pegar o tamanho definido população de matrizes de horários
def criarHorarios(n):
    quantidades = {
        "AG": 7,
        "PI1": 1,
        "GA": 4,
        "EE": 3,
        "IE": 2,
        "CA": 5,
        "ILM": 2,
        "Vaga": 6  # Adicionando aulas vagas para completar 30 aulas
    }
    linhas = 5
    colunas = 6
    total_aulas = sum(quantidades.values())
    if total_aulas != linhas * colunas:
        raise ValueError("A quantidade total de aulas não corresponde ao número de células na matriz")
    horarios = []
    for _ in range(n):
        todas_aulas = []
        for aula, quantidade in quantidades.items():
            todas_aulas.extend([aula] * quantidade)

        random.shuffle(todas_aulas)
        matriz = []
        index = 0
        for i in range(linhas):
            linha = []
            for j in range(colunas):
                linha.append(todas_aulas[index])
                index += 1
            matriz.append(linha)
        horarios.append(matriz)
    return horarios

def calcularFitness(horarios):
    # Quantidade esperada de cada aula
    quantidades_esperadas = {
        "AG": 7,
        "PI1": 1,
        "GA": 4,
        "EE": 3,
        "IE": 2,
        "CA": 5,
        "ILM": 2,
        "Vaga": 6
    }

    fitness = []
    for horario in horarios:
        contagem = defaultdict(int)
        for linha in horario:
            for aula in linha:
                contagem[aula] += 1
        
        # Verificar se a contagem está correta
        fitness_score = 0
        if all(contagem[aula] == quantidade for aula, quantidade in quantidades_esperadas.items()):
            fitness_score += 10
        
        # Verificar aulas duplas, triplas ou quádruplas
        for linha in horario:
            consecutivos = 1
            for j in range(1, len(linha)):
                
                if linha[j] != "Vaga": 
                    if linha[j] == linha[j - 1]:
                        consecutivos += 1
                    else:
                        consecutivos = 1
                
                if consecutivos == 2:
                    fitness_score += 10 # Aula dupla
                elif consecutivos == 3:
                    fitness_score -= 5  # Três aulas seguidas
                elif consecutivos == 4:
                    fitness_score -= 10  # Quatro aulas seguidas
                    
        # Verificar localização das aulas vagas
        for linha in horario:
            if linha[0] == 'Vaga':
                fitness_score += 200  # Aumentar a recompensa para vaga no início da linha
            if linha[-1] == 'Vaga':
                fitness_score += 200  # Aumentar a recompensa para vaga no final da linha
            for j in range(1, len(linha) - 1):
                if linha[j] == 'Vaga':
                    fitness_score -= 50  # Aumentar a penalização para vaga no meio da linha

        fitness.append(fitness_score)
    return fitness


def selecao_torneio(tamanho_torneio, fitness, horarios):
    selecionados = []
    populacao_tamanho = len(horarios)
    while len(selecionados) < populacao_tamanho:
        grupo_indices = random.sample(range(populacao_tamanho), tamanho_torneio)
        grupo_fitness = [fitness[i] for i in grupo_indices]
        grupo_horarios = [horarios[i] for i in grupo_indices]
        grupo_ordenado = sorted(zip(grupo_fitness, grupo_horarios), key=lambda x: x[0], reverse=True)
        selecionados.append(grupo_ordenado[0][1])
        selecionados.append(grupo_ordenado[1][1])
    return selecionados[:populacao_tamanho]

def separarMelhores(fitness, horarios, n):
    combinados = list(zip(fitness, horarios))
    combinados_ordenados = sorted(combinados, key=lambda x: x[0], reverse=True)
    melhores_horarios = [horario for _, horario in combinados_ordenados[:n]]
    return melhores_horarios

def sortearCasais(qtd_melhores):
    """Sorteia casais para cruzamento."""
    quantidade_casais = int(qtd_melhores/2)
    casais_sorteados = np.empty((quantidade_casais, 2), dtype=int)
    numeros_possiveis = np.arange(qtd_melhores)
    for i in range(quantidade_casais):
        indice_numero1 = np.random.choice(numeros_possiveis)
        numeros_possiveis = numeros_possiveis[numeros_possiveis != indice_numero1]
        indice_numero2 = np.random.choice(numeros_possiveis)
        casais_sorteados[i] = [indice_numero1, indice_numero2]
    return casais_sorteados

def gerarPontoCorte(tamanho):
    return rd.randint(1, tamanho - 1)

def recombinar(ponto_corte, casais, melhores_horarios):
    """Realiza o cruzamento entre os pais."""
    filhos = []
    for casal in casais:
        #print("\n\n\n")
        #print(melhores_horarios[0])
        pai1_idx, pai2_idx = casal
        pai1 = melhores_horarios[casal[0]]
        pai2 = melhores_horarios[casal[1]]
        filho1 = []
        filho2 = []
        # Realiza o crossover na posição do ponto de corte para cada linha
        for i in range(len(pai1)):
            linha_pai1 = pai1[i]
            linha_pai2 = pai2[i]
            filho1.append(linha_pai1[:ponto_corte] + linha_pai2[ponto_corte:])
            filho2.append(linha_pai2[:ponto_corte] + linha_pai1[ponto_corte:])
        filhos.append(filho1)
        filhos.append(filho2)
    return filhos

def gerarPontoCorte2(tamanho_coluna , tamanho_linha):
    random1 = rd.randint(1, tamanho_coluna - 1)
    random2 = rd.randint(1, tamanho_linha - 1)
    return random1,random2

def recombinar2(pontos_corte, casais, melhores_horarios):
    """Realiza o cruzamento de dois pontos entre os pais."""
    filhos = []
    ponto_corte_coluna, ponto_corte_linha = pontos_corte
    
    for casal in casais:
        #pai1_idx, pai2_idx = casal
        
        pai1 = melhores_horarios[casal[0]]
        pai2 = melhores_horarios[casal[1]]
        filho1 = [linha[:] for linha in pai1]
        filho2 = [linha[:] for linha in pai2]  
        
        for i in range(ponto_corte_linha, len(pai1)):
            filho1[i][:ponto_corte_coluna], filho2[i][:ponto_corte_coluna] = pai2[i][:ponto_corte_coluna], pai1[i][:ponto_corte_coluna]
        filhos.append(filho1)
        filhos.append(filho2)
    return filhos

def gerarMutacao(horarios, taxa_mutacao, tamanho_horario):
    """Realiza a mutação dos filhos."""
    total_elementos = tamanho_horario[0] * tamanho_horario[1]
    num_mutacoes = int(total_elementos * (taxa_mutacao / 100))
    for horario in horarios:
        for _ in range(num_mutacoes):
            # Seleciona aleatoriamente duas posições diferentes para trocar os elementos
            linha1, coluna1 = random.randint(0, tamanho_horario[0] - 1), random.randint(0, tamanho_horario[1] - 1)
            linha2, coluna2 = random.randint(0, tamanho_horario[0] - 1), random.randint(0, tamanho_horario[1] - 1)
            # Troca os elementos nas posições selecionadas
            horario[linha1][coluna1], horario[linha2][coluna2] = horario[linha2][coluna2], horario[linha1][coluna1]
    return horarios

def separarPiores(fitness, horarios, n):
    """Separa os n piores elementos."""
    # Combina fitness e horários em uma única lista
    combinados = list(zip(fitness, horarios))
    combinados_ordenados = sorted(combinados, key=lambda x: x[0])
    piores_horarios = [horario for _, horario in combinados_ordenados[:n]]
    return piores_horarios

def substituirPioresPorFilhos(populacao, piores, filhos):
    """Substitui os piores elementos pelos filhos."""
    piores_indices = [populacao.index(pior) for pior in piores]
    # Substitui os piores pelos filhos nas posições correspondentes
    for i, indice in enumerate(piores_indices):
        populacao[indice] = filhos[i]
    return populacao

def misturarNovosAntigos(melhores_horarios, filhos, tamanho_populacao):
    """Combina os melhores elementos com os filhos."""
    # Junta os melhores horários e os filhos em uma única lista
    nova_populacao = melhores_horarios + filhos
    # Trunca a nova população para manter o tamanho original
    nova_populacao = nova_populacao[:tamanho_populacao]
    return nova_populacao

#separar os melhores pelo fitness - cruzamento de matrizes
#subatituir os de piores fitness pelos filhos
#pegar o maior fitness, a posição dele vai ser a posição do melhor horario
melhor_horario_geral = None
melhor_fitness_geral = 0
for aux in range(geracoes):
    populacao = criarHorarios(tamanho_populacao)
    pop_fitness = calcularFitness(populacao)
    if torneio_roleta:
        melhores = selecao_torneio(tamanho_torneio, pop_fitness, populacao)
    else:
        melhores = separarMelhores(pop_fitness, populacao, qtd_melhores)
    #print(melhores)
    #print("\n\n\n\n")
    recomb = sortearCasais(qtd_melhores)
    #print(recomb)
    if pontos:
        pontoCorte = gerarPontoCorte(tamanho_coluna)
        filhos = recombinar(pontoCorte, recomb, melhores)
    else:
        pontosCorte = gerarPontoCorte2(tamanho_coluna, tamanho_linha)
        filhos = recombinar2(pontosCorte, recomb, melhores)
    filhos = gerarMutacao(filhos, taxa_mutacao, tamanho_horario)
    piores_elementos = separarPiores(pop_fitness, populacao, qtd_melhores)
    populacao = substituirPioresPorFilhos(populacao, piores_elementos, filhos)
    print("\n Geração: ", aux)
    best_fitness =  max(pop_fitness)
    print(" Maior Fitness: ", best_fitness)
    indice_melhor_g = pop_fitness.index(best_fitness)
    melhor_horario_g = populacao[indice_melhor_g]

    # Atualizar o melhor horário geral
    if best_fitness > melhor_fitness_geral:
        melhor_fitness_geral = best_fitness
        melhor_horario_geral = melhor_horario_g
    
    aux = aux + 1
    
print("\n\nMelhor horário geral (maior fitness):")
print(np.array(melhor_horario_geral))
print(f"Fitness: {melhor_fitness_geral}")