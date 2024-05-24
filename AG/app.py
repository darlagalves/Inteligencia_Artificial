# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:07:31 2024

@author: darla

O código recebe por input da tela feita pela biblioteca streamlit os valores tamanho do cromossomo, tamanho da população, quantidade de melhores, probabilidade de mutação e quantidade de gerações
depois passa esses valores para o algoritimo genético para os cálculos, depois disso é imprimido da tela os resultados de cada geração e o gráfio da função com o ponto de mínimo global
"""
import streamlit as st
import numpy as np
import panda as pd
import random as rd
import matplotlib.pyplot as plt
import math
from collections import Counter

st.title("Algoritmo Genético")


# Funções do algoritmo genético (mesmas do código anterior)
def gerarElementos(n):
    vetor = [rd.randint(0, 512) for _ in range(n)]
    return vetor

def gerarElementosBinarios(vetor_aleatorio, tamanho_cromossomo):
    vetor_binario = []
    for numero in vetor_aleatorio:
        numero_binario_str = f"{numero:0>{tamanho_cromossomo}b}"
        vetor_binario.append(numero_binario_str)
    return vetor_binario

def gerarImagem(vetor_aleatorio):
    vetor_imagem = np.empty_like(vetor_aleatorio)
    for i in range(len(vetor_aleatorio)):
        valor_imagem = -abs(vetor_aleatorio[i] * math.sin(math.sqrt(abs(vetor_aleatorio[i]))))
        vetor_imagem[i] = valor_imagem
    return vetor_imagem

def gerarProbabilidades(vetor_imagem):
    vetor_prob = (vetor_imagem / (sum(vetor_imagem))) * 100
    return vetor_prob

def separarMelhores(vetor_prob, vetor_binario, n):
    porcentagem_com_indices = list(enumerate(vetor_prob))
    porcentagem_com_indices.sort(key=lambda x: x[1], reverse=True)
    indices_melhores = [indice for indice, porcentagem in porcentagem_com_indices[:n]]
    vetor_melhores = [vetor_binario[indice] for indice in indices_melhores]
    return vetor_melhores

def sortearCasais(qtd_melhores):
    quantidade_casais = int(qtd_melhores/2)
    casais_sorteados = np.empty((quantidade_casais, 2), dtype=int)
    numeros_possiveis = np.arange(qtd_melhores)
    for i in range(quantidade_casais):
        indice_numero1 = np.random.choice(numeros_possiveis)
        numeros_possiveis = numeros_possiveis[numeros_possiveis != indice_numero1]
        indice_numero2 = np.random.choice(numeros_possiveis)
        casais_sorteados[i] = [indice_numero1, indice_numero2]
    return casais_sorteados

def gerarPontoCorte():
    numero_sorteado = rd.randint(1, 9)
    return numero_sorteado

def recombinar(numero_sorteado, casais_sorteados, vetor_melhores):
    filhos = []
    for i, casal in enumerate(casais_sorteados):
        pai1 = vetor_melhores[casal[0]]
        pai2 = vetor_melhores[casal[1]]
        filho1 = pai1[:numero_sorteado] + pai2[numero_sorteado:]
        filho2 = pai2[:numero_sorteado] + pai1[numero_sorteado:]
        filhos.append(filho1)
        filhos.append(filho2)
    return filhos

def gerarMutacao(filhos, prob_mutacao):
    num_mutacao = prob_mutacao/100
    num_filhos_mutados = int(len(filhos) * num_mutacao)
    indices_mutados = rd.sample(range(len(filhos)), num_filhos_mutados)
    vetor_modificado = filhos.copy()
    for indice in indices_mutados:
        posicao = rd.randint(0, 9)
        lista_binario = list(vetor_modificado[indice])
        lista_binario[posicao] = '0' if lista_binario[posicao] == '1' else '1'
        binario_modificado = ''.join(lista_binario)
        vetor_modificado[indice] = binario_modificado
    return vetor_modificado

def separarPiores(vetor_prob, vetor_binario, n):
    porcentagem_com_indices = list(enumerate(vetor_prob))
    porcentagem_com_indices.sort(key=lambda x: x[1])
    indices_piores = [indice for indice, porcentagem in porcentagem_com_indices[:n]]
    vetor_piores = [vetor_binario[indice] for indice in indices_piores]
    return vetor_piores

def substituirPioresPorFilhos(populacao, piores_elementos, filhos_melhores):
    if len(piores_elementos) != len(filhos_melhores):
        raise ValueError("O número de piores elementos deve ser igual ao número de filhos dos melhores")
    indices_piores = [populacao.index(elemento) for elemento in piores_elementos]
    for indice, filho in zip(indices_piores, filhos_melhores):
        populacao[indice] = filho
    return populacao

def misturarNovosAntigos(vetor_binario_melhores, filhos):
    vetor_combinado = vetor_binario_melhores + filhos
    return vetor_combinado

def novosValoresDecimais(vetor_binarios):
    vetor_decimais = []
    for numero_binario in vetor_binarios:
        valor_decimal = int(numero_binario, 2)
        vetor_decimais.append(valor_decimal)
    return vetor_decimais

# Recebendo os valores do usuário
col1, space, col2 = st.columns([1, 0.2, 1])  # Define as larguras das colunas

with col1:
    tamanho_cromossomo = st.number_input("Tamanho do Cromossomo", min_value=1, value=10)
    tamanho_populacao = st.number_input("Tamanho da População", min_value=1, value=100)
    qtd_melhores = st.number_input("Quantidade de Melhores por Geração", min_value=1, value=50)
    prob_mutacao = st.number_input("Probabilidade de Mutação (em porcentagem)", min_value=0, max_value=100, value=10)
    qtd_geracoes = st.number_input("Quantidade de Gerações", min_value=1, value=100)

with space:
    st.empty()

with col2:
    if st.button("Executar"):
        popDec = gerarElementos(tamanho_populacao)
        popBin = gerarElementosBinarios(popDec, tamanho_cromossomo)
        imagemFuncao = gerarImagem(popDec)
        probabRolet = gerarProbabilidades(imagemFuncao)
        sMelhores = separarMelhores(probabRolet, popBin, qtd_melhores)
        recomb = sortearCasais(qtd_melhores)
        pontoCorte = gerarPontoCorte()
        s_filhos = recombinar(pontoCorte, recomb, sMelhores)
        s_filhos = gerarMutacao(s_filhos, prob_mutacao)
        sPiores = separarPiores(probabRolet, popBin, qtd_melhores)
        popBin = substituirPioresPorFilhos(popBin, sPiores, s_filhos)
        popDec = novosValoresDecimais(popBin)
        chart_data = pd.DataFrame(columns=['x', 'y'])
        # Laço de gerações
        for aux in range(qtd_geracoes):
            imagemFuncao = gerarImagem(popDec)
            probabRolet = gerarProbabilidades(imagemFuncao)
            sMelhores = separarMelhores(probabRolet, popBin, qtd_melhores)
            recomb = sortearCasais(qtd_melhores)
            pontoCorte = gerarPontoCorte()
            s_filhos = recombinar(pontoCorte, recomb, sMelhores)
            s_filhos = gerarMutacao(s_filhos, prob_mutacao)
            sPiores = separarPiores(probabRolet, popBin, qtd_melhores)
            popBin = substituirPioresPorFilhos(popBin, sPiores, s_filhos)
            popDec = novosValoresDecimais(popBin)

            # Análise da população
            contador = Counter(popDec)
            elemento_mais_frequente, frequencia_maxima = contador.most_common(1)[0]
            # Adicionando dados ao DataFrame
            chart_data = chart_data.append({'x': aux + 1, 'y': frequencia_maxima}, ignore_index=True)
            #st.write(f"Geração {aux + 1}:")
            #st.write(f"Número mais frequente: {elemento_mais_frequente}, Frequência: {frequencia_maxima}")

        # Plotando o gráfico da frequência máxima
        st.line_chart(chart_data, title='Frequência Máxima por Geração', use_container_width=True, y_axis_format="%.2f")

        # Plotando o gráfico da função (sem o ponto)
        st.line_chart(pd.DataFrame({
            'x': np.linspace(0, 1024, 1000),
            'y': -abs(np.linspace(0, 1024, 1000) * np.sin(np.sqrt(abs(np.linspace(0, 1024, 1000)))))
        }), title='-abs(x * sin(sqrt(abs(x))))', use_container_width=True, y_axis_format="%.2f")

        # Marcando o ponto do elemento mais frequente
        st.pyplot(plt.figure(figsize=(15, 12)))
        y_elemento_mais_frequente = -abs(elemento_mais_frequente * np.sin(np.sqrt(abs(elemento_mais_frequente))))
        plt.scatter([elemento_mais_frequente], [y_elemento_mais_frequente], color='red')
        plt.text(elemento_mais_frequente, y_elemento_mais_frequente, f'({elemento_mais_frequente}, {y_elemento_mais_frequente:.2f})',
                 fontsize=12, ha='right')
        st.pyplot(plt) 

        # Imprimindo as gerações em um bloco expansível
        with st.expander("Resultados por Geração"):
            for aux in range(qtd_geracoes):
                contador = Counter(popDec)
                elemento_mais_frequente, frequencia_maxima = contador.most_common(1)[0]
                st.write(f"Geração {aux + 1}: Número mais frequente: {elemento_mais_frequente}, Frequência: {frequencia_maxima}")
