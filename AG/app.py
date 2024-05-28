# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:07:31 2024

@author: darla

O código recebe por input da tela feita pela biblioteca streamlit os valores tamanho do cromossomo, tamanho da população, quantidade de melhores, probabilidade de mutação e quantidade de gerações
depois passa esses valores para o algoritimo genético para os cálculos, depois disso é imprimido da tela os resultados de cada geração e o gráfio da função com o ponto de mínimo global
"""
import streamlit as st
import numpy as np
import pandas as pd
import requests as rq
import random as rd
import matplotlib.pyplot as plt
import math
import altair as alt
from collections import Counter

st.title("Algoritmo Genético")


# Funções do algoritmo genético (mesmas do código anterior)
def gerarElementos(n, limite_inferior=0, limite_superior=512):
    """Gera um vetor de n números aleatórios com virgula entre os limites inferior e superior."""
    vetor = [rd.uniform(limite_inferior, limite_superior) for _ in range(n)]
    return vetor

def gerarElementosBinarios(vetor_aleatorio, tamanho_cromossomo, precisao):
    """Converte números decimais para binários."""
    vetor_binario = []
    for numero in vetor_aleatorio:
        # Calcula o inteiro e a fração do número
        inteiro = int(numero)
        fracao = numero - inteiro
        
        # Converte o inteiro para binário
        inteiro_binario = f"{inteiro:0>{tamanho_cromossomo}b}"
        
        # Converte a fração para binário
        fracao_binaria = ''
        for _ in range(precisao):
            fracao *= 2
            if fracao >= 1:
                fracao_binaria += '1'
                fracao -= 1
            else:
                fracao_binaria += '0'
        
        # Combina o inteiro e a fração binários
        numero_binario_str = inteiro_binario + '.' + fracao_binaria
        vetor_binario.append(numero_binario_str)
    return vetor_binario

def gerarImagem(vetor_aleatorio):
    """Calcula a função para cada elemento do vetor."""
    vetor_imagem = np.empty_like(vetor_aleatorio)
    for i in range(len(vetor_aleatorio)):
        valor_imagem = -abs(vetor_aleatorio[i] * math.sin(math.sqrt(abs(vetor_aleatorio[i]))))
        vetor_imagem[i] = valor_imagem
    return vetor_imagem

def gerarProbabilidades(vetor_imagem):
    """Calcula a probabilidade de cada elemento."""
    vetor_prob = (vetor_imagem / (sum(vetor_imagem))) * 100
    return vetor_prob

def selecao_torneio(vetor_prob, vetor_binario, tamanho_torneio):
    """Realiza a seleção por torneio."""
    indices_torneio = np.random.choice(range(len(vetor_prob)), size=tamanho_torneio, replace=False)
    melhores_torneio = [(vetor_binario[i], vetor_prob[i]) for i in indices_torneio]
    melhores_torneio.sort(key=lambda x: x[1], reverse=True)
    melhor_torneio = melhores_torneio[0][0]
    return melhor_torneio

def separarMelhores(vetor_prob, vetor_binario, n):
    """Separa os n melhores elementos."""
    porcentagem_com_indices = list(enumerate(vetor_prob))
    porcentagem_com_indices.sort(key=lambda x: x[1], reverse=True)
    indices_melhores = [indice for indice, porcentagem in porcentagem_com_indices[:n]]
    vetor_melhores = [vetor_binario[indice] for indice in indices_melhores]
    return vetor_melhores

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

def gerarPontoCorte(tamanho_cromossomo):
    return rd.randint(1, tamanho_cromossomo - 1)

def recombinar(numero_sorteado, casais_sorteados, vetor_melhores):
    """Realiza o cruzamento entre os pais."""
    filhos = []
    for i, casal in enumerate(casais_sorteados):
        pai1 = vetor_melhores[casal[0]]
        pai2 = vetor_melhores[casal[1]]
        # Divide os pais no ponto de corte
        pai1_partes = pai1.split('.')
        pai2_partes = pai2.split('.')
        # Recombina as partes dos pais
        filho1 = pai1_partes[0][:numero_sorteado] + pai2_partes[0][numero_sorteado:] + '.' + pai1_partes[1]
        filho2 = pai2_partes[0][:numero_sorteado] + pai1_partes[0][numero_sorteado:] + '.' + pai2_partes[1]
        filhos.append(filho1)
        filhos.append(filho2)
    return filhos

def gerarPontoCorte2(tamanho_cromossomo):
    if (tamanho_cromossomo <= 12):
        random1 = rd.randint(1, tamanho_cromossomo - 6)
        random2 = rd.randint(tamanho_cromossomo - 4, tamanho_cromossomo - 1)
    else:
        random1 = rd.randint(1, tamanho_cromossomo - -10)
        random2 = rd.randint(tamanho_cromossomo - 6, tamanho_cromossomo - 1)
    return random1,random2

def recombinar2(pontos_corte, casais_sorteados, vetor_melhores):
    """Realiza o cruzamento de dois pontos entre os pais."""
    filhos = []
    for casal in casais_sorteados:
        pai1 = vetor_melhores[casal[0]]
        pai2 = vetor_melhores[casal[1]]
        corte1, corte2 = sorted(pontos_corte)
        pai1_partes = pai1.split('.')
        pai2_partes = pai2.split('.')
        filho1 = pai1_partes[0][:corte1] + pai2_partes[0][corte1:corte2] + pai1_partes[0][corte2:] + '.' + pai1_partes[1]
        filho2 = pai2_partes[0][:corte1] + pai1_partes[0][corte1:corte2] + pai2_partes[0][corte2:] + '.' + pai2_partes[1]
        filhos.append(filho1)
        filhos.append(filho2)
    return filhos

def gerarMutacao(filhos, prob_mutacao, tamanho_cromossomo):
    """Realiza a mutação dos filhos."""
    num_mutacao = prob_mutacao/100
    num_filhos_mutados = int(len(filhos) * num_mutacao)
    indices_mutados = rd.sample(range(len(filhos)), num_filhos_mutados)
    vetor_modificado = filhos.copy()
    for indice in indices_mutados:
        # Realiza a mutação com base no tamanho do cromossomo
        posicao = rd.randint(0, tamanho_cromossomo-1)
        lista_binario = list(vetor_modificado[indice])
        lista_binario[posicao] = '0' if lista_binario[posicao] == '1' else '1'
        binario_modificado = ''.join(lista_binario)
        vetor_modificado[indice] = binario_modificado
    return vetor_modificado

def separarPiores(vetor_prob, vetor_binario, n):
    """Separa os n piores elementos."""
    porcentagem_com_indices = list(enumerate(vetor_prob))
    porcentagem_com_indices.sort(key=lambda x: x[1])
    indices_piores = [indice for indice, porcentagem in porcentagem_com_indices[:n]]
    vetor_piores = [vetor_binario[indice] for indice in indices_piores]
    return vetor_piores

def substituirPioresPorFilhos(populacao, piores_elementos, filhos_melhores):
    """Substitui os piores elementos pelos filhos."""
    if len(piores_elementos) != len(filhos_melhores):
        raise ValueError("O número de piores elementos deve ser igual ao número de filhos dos melhores")
    indices_piores = [populacao.index(elemento) for elemento in piores_elementos]
    for indice, filho in zip(indices_piores, filhos_melhores):
        populacao[indice] = filho
    return populacao

def misturarNovosAntigos(vetor_binario_melhores, filhos):
    """Combina os melhores elementos com os filhos."""
    vetor_combinado = vetor_binario_melhores + filhos
    return vetor_combinado

def novosValoresDecimais(vetor_binarios):
    """Converte números binários para decimais."""
    vetor_decimais = []
    for numero_binario in vetor_binarios:
        # Divide o número binário em partes inteira e fracionária
        partes = numero_binario.split('.')
        inteiro = int(partes[0], 2)
        fracao = 0
        if len(partes) > 1:
            for i, bit in enumerate(partes[1]):
                fracao += int(bit) * (2 ** -(i + 1))
        valor_decimal = inteiro + fracao
        vetor_decimais.append(valor_decimal)
    return vetor_decimais

def elitism(vetor_prob, vetor_binario, n_elites):
    """Seleciona os n_elites melhores indivíduos com base em sua aptidão."""
    # Combine as representações binárias e suas probabilidades
    pop_com_fitness = list(zip(vetor_binario, vetor_prob))
    # Ordene com base na aptidão
    pop_com_fitness.sort(key=lambda x: x[1], reverse=True)
    # Selecione os elitistas
    elitistas = [ind for ind, fit in pop_com_fitness[:n_elites]]
    return elitistas

# Recebendo os valores do usuário
col1, space, col2 = st.columns([1, 0.2, 1])  # Define as larguras das colunas

with col1:
    tamanho_cromossomo = st.number_input("Tamanho do Cromossomo (parte inteira)", min_value=1, value=10)
    tamanho_populacao = st.number_input("Tamanho da População", min_value=1, value=100)
    qtd_melhores = st.number_input("Quantidade de Melhores por Geração", min_value=1, value=50)
    prob_mutacao = st.number_input("Probabilidade de Mutação (em porcentagem)", min_value=0, max_value=100, value=10)
    qtd_geracoes = st.number_input("Quantidade de Gerações", min_value=1, value=100)
    precisao_fracao = st.number_input("Precisão da Parte Fracionária (bits)", min_value=1, value=8)
with space:
    st.empty()

with col2:
    # Define um estado para o checkbox
    switch_state_roleta = st.session_state.get('switch_state_roleta', False)
    switch_state_torneio = st.session_state.get('switch_state_torneio', False)
    switch_state_um_ponto = st.session_state.get('switch_state_um_ponto', False)
    switch_state_dois_pontos = st.session_state.get('switch_state_dois_pontos', False)
    # Crie o checkbox
    switch_state_roleta = st.checkbox("Roleta", switch_state_roleta, key="switch1")
    switch_state_torneio = st.checkbox("Torneio", switch_state_torneio, key="switch2")
    
    # Imprima o estado do checkbox
    if(switch_state_torneio):
        tamanho_torneio = st.number_input("Tamanho do Torneio", min_value=1, value=10)
           
    switch_state_um_ponto = st.checkbox("Cruzamento de 1 Ponto", switch_state_um_ponto, key="switch3")
    switch_state_dois_pontos = st.checkbox("Cruzamento de 2 Pontos", switch_state_dois_pontos, key="switch4")
    tamanho_elite = st.number_input("Tamanho do Elitismo", min_value=0, value=1)

if st.button("Executar"):
    popDec = gerarElementos(tamanho_populacao)
    popBin = gerarElementosBinarios(popDec, tamanho_cromossomo, precisao_fracao)
    imagemFuncao = gerarImagem(popDec)
    probabRolet = gerarProbabilidades(imagemFuncao)
    elites = elitism(probabRolet, popBin, tamanho_elite)
    popBin[:tamanho_elite] = elites

    chart_data = pd.DataFrame(columns=['x', 'y'])

    with st.expander("Resultados por Geração"):
        for aux in range(qtd_geracoes):
            popBin[:tamanho_elite] = elites
            if switch_state_roleta:
                sMelhores = separarMelhores(probabRolet, popBin, qtd_melhores)
            elif switch_state_torneio:
                sMelhores = [selecao_torneio(probabRolet, popBin, tamanho_torneio) for _ in range(qtd_melhores)]

            recomb = sortearCasais(qtd_melhores)
            if switch_state_um_ponto:
                pontoCorte = gerarPontoCorte(tamanho_cromossomo)
                s_filhos = recombinar(pontoCorte, recomb, sMelhores)
            elif switch_state_dois_pontos:
                pontosCorte = gerarPontoCorte2(tamanho_cromossomo)
                s_filhos = recombinar2(pontosCorte, recomb, sMelhores)
            s_filhos = gerarMutacao(s_filhos, prob_mutacao, tamanho_cromossomo)
            # Substituir os piores indivíduos pelos filhos
            sPiores = separarPiores(probabRolet, popBin, qtd_melhores)
            popBin = substituirPioresPorFilhos(popBin, sPiores, s_filhos)
            popDec = novosValoresDecimais(popBin)

            imagemFuncao = gerarImagem(popDec)
            probabRolet = gerarProbabilidades(imagemFuncao)

            contador = Counter(popDec)
            elemento_mais_frequente, frequencia_maxima = contador.most_common(1)[0]
            chart_data = pd.concat([chart_data, pd.DataFrame({'x': [aux + 1], 'y': [frequencia_maxima]})], ignore_index=True)

            st.write(f"Geração {aux + 1}: Número mais frequente: {elemento_mais_frequente}, Frequência: {frequencia_maxima}")

    # Visualizando a função de fitness
    x_limit = 2 ** tamanho_cromossomo
    x_values = np.linspace(0, x_limit, 1000)
    y_values = -abs(x_values * np.sin(np.sqrt(abs(x_values))))
    y_elemento_mais_frequente = -abs(elemento_mais_frequente * np.sin(np.sqrt(abs(elemento_mais_frequente))))

    chart = alt.Chart(pd.DataFrame({'x': x_values, 'y': y_values})).mark_line().encode(
        x='x',
        y='y'
    ).properties(
        title='-abs(x * sin(sqrt(abs(x))))'
    )
    chart = chart + alt.Chart(pd.DataFrame({'x': [elemento_mais_frequente], 'y': [y_elemento_mais_frequente]})).mark_point(color='red').encode(
        x='x',
        y='y'
    )

    st.altair_chart(chart, use_container_width=True)