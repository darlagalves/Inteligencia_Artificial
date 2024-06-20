import streamlit as st
import numpy as np
import pandas as pd
import random as rd
import math
import altair as alt
import plotly.graph_objects as go

st.title("Algoritmo Genético")

# Funções do algoritmo genético
def gerarElementos(n):
    """Gera um vetor de n números aleatórios com virgula entre os limites inferior e superior."""
    limite_inferior_x = -3.1
    limite_superior_x = 12.1
    limite_inferior_y = 4.1
    limite_superior_y = 5.8
    vetor = [[rd.uniform(limite_inferior_x, limite_superior_x), rd.uniform(limite_inferior_y, limite_superior_y)] for _ in range(n)]
    return vetor

def gerarImagem(vetor_aleatorio):
    """Calcula a função para cada elemento do vetor."""
    vetor_imagem = []
    for elemento in vetor_aleatorio:
        valor_imagem = 15 + (elemento[0] *math.cos(2 * math.pi * elemento[0])) + (elemento[1] * math.cos(14 * math.pi * elemento[1]))
        vetor_imagem.append(valor_imagem)
    return np.array(vetor_imagem)

def gerarProbabilidades(vetor_imagem):
    """Calcula a probabilidade de cada elemento."""
    vetor_prob = (vetor_imagem / (sum(vetor_imagem))) * 100
    return vetor_prob

def selecao_torneio(vetor_prob, vetor, tamanho_torneio):
    """Realiza a seleção por torneio."""
    indices_torneio = np.random.choice(range(len(vetor_prob)), size=tamanho_torneio, replace=False)
    melhores_torneio = [(vetor[i], vetor_prob[i]) for i in indices_torneio]
    melhores_torneio.sort(key=lambda x: x[1], reverse=True)
    melhor_torneio = melhores_torneio[0][0]
    return melhor_torneio

def separarMelhores(vetor_prob, vetor, n):
    """Separa os n melhores elementos."""
    porcentagem_com_indices = list(enumerate(vetor_prob))
    porcentagem_com_indices.sort(key=lambda x: x[1], reverse=True)
    indices_melhores = [indice for indice, porcentagem in porcentagem_com_indices[:n]]
    vetor_melhores = [vetor[indice] for indice in indices_melhores]
    return vetor_melhores

def sortearCasais(qtd_melhores):
    """Sorteia casais para cruzamento."""
    quantidade_casais = int(qtd_melhores / 2)
    casais_sorteados = np.empty((quantidade_casais, 2), dtype=int)
    numeros_possiveis = np.arange(qtd_melhores)
    for i in range(quantidade_casais):
        indice_numero1 = np.random.choice(numeros_possiveis)
        numeros_possiveis = numeros_possiveis[numeros_possiveis != indice_numero1]
        indice_numero2 = np.random.choice(numeros_possiveis)
        casais_sorteados[i] = [indice_numero1, indice_numero2]
    return casais_sorteados

def recombinar(casais_sorteados, vetor_melhores):
    """Realiza o cruzamento entre os pais."""
    beta = rd.randint(0, 1)
    filhos = []
    for indice_melhor, casal in enumerate(casais_sorteados):
        melhor = vetor_melhores[indice_melhor]  
        pai1 = vetor_melhores[casal[0]]
        pai2 = vetor_melhores[casal[1]]
        elemento_1_1 = beta * pai1[0] + (1 - beta) * pai2[0]
        elemento_2_1 = beta * pai1[1] + (1 - beta) * pai2[1]
        filho1 = [elemento_1_1, elemento_2_1]
        elemento_1_2 = (1 - beta) * pai1[0] + beta * pai2[0]
        elemento_2_2 = (1 - beta) * pai1[1] + beta * pai2[1]
        filho2 = [elemento_1_2, elemento_2_2]
        filhos.append(filho1)
        filhos.append(filho2)
    return filhos

def gerarMutacao(filhos, prob_mutacao):
    """Realiza a mutação dos filhos."""
    num_mutacao = prob_mutacao / 100
    num_filhos_mutados = int(len(filhos) * num_mutacao)
    indices_mutados = rd.sample(range(len(filhos)), num_filhos_mutados)
    vetor_modificado = filhos.copy()
    limite_inferior_x = -3.1
    limite_superior_x = 12.1
    limite_inferior_y = 4.1
    limite_superior_y = 5.8
    for indice in indices_mutados:
        posicao = rd.choice([0, 1])
        if posicao == 0:
            vetor_modificado[indice][posicao] = rd.uniform(limite_inferior_x, limite_superior_x)
        else:
            vetor_modificado[indice][posicao] = rd.uniform(limite_inferior_y, limite_superior_y)
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

def elitism(vetor_prob, vetor_binario, n_elites):
    """Seleciona os n_elites melhores indivíduos com base em sua aptidão."""
    pop_com_fitness = list(zip(vetor_binario, vetor_prob))
    pop_com_fitness.sort(key=lambda x: x[1], reverse=True)
    elitistas = [ind for ind, fit in pop_com_fitness[:n_elites]]
    return elitistas

# Recebendo os valores do usuário
col1, space, col2 = st.columns([1, 0.2, 1])  # Define as larguras das colunas

with col1:
    tamanho_populacao = st.number_input("Tamanho da População", min_value=1, value=100)
    qtd_melhores = st.number_input("Quantidade de Melhores por Geração", min_value=1, value=50)
    prob_mutacao = st.number_input("Probabilidade de Mutação (em porcentagem)", min_value=0, max_value=100, value=10)
    qtd_geracoes = st.number_input("Quantidade de Gerações", min_value=1, value=100)
with space:
    st.empty()

with col2:
    switch_state_roleta = st.session_state.get('switch_state_roleta', False)
    switch_state_torneio = st.session_state.get('switch_state_torneio', False)
    switch_state_roleta = st.checkbox("Roleta", switch_state_roleta, key="switch1")
    switch_state_torneio = st.checkbox("Torneio", switch_state_torneio, key="switch2")
    
    if switch_state_torneio:
        tamanho_torneio = st.number_input("Tamanho do Torneio", min_value=1, value=10)
           
    tamanho_elite = st.number_input("Tamanho do Elitismo", min_value=0, value=1)

if st.button("Executar"):
    popDec = gerarElementos(tamanho_populacao)
    imagemFuncao = gerarImagem(popDec)
    probabRolet = gerarProbabilidades(imagemFuncao)
    valores_fitness_geracao = []
    populacoes = []

    with st.expander("Resultados por Geração"):
        for aux in range(qtd_geracoes):
            if switch_state_roleta:
                sMelhores = separarMelhores(probabRolet, popDec, qtd_melhores)
            elif switch_state_torneio:
                sMelhores = [selecao_torneio(probabRolet, popDec, tamanho_torneio) for _ in range(qtd_melhores)]

            recomb = sortearCasais(qtd_melhores)
            s_filhos = recombinar(recomb, sMelhores)
            s_filhos = gerarMutacao(s_filhos, prob_mutacao)
            sPiores = separarPiores(probabRolet, popDec, qtd_melhores)
            popDec = substituirPioresPorFilhos(popDec, sPiores, s_filhos)

            imagemFuncao = gerarImagem(popDec)
            probabRolet = gerarProbabilidades(imagemFuncao)

            # Armazenando valores de fitness para impressão
            valores_fitness = gerarImagem(popDec)
            valores_fitness_geracao.append((aux, popDec))
            # Armazenando as populações para o gráfico
            for individuo in popDec:
                populacoes.append({'geracao': aux, 'x': individuo[0], 'y': individuo[1]})

    # Criando o dropdown para os valores de fitness
    st.write("Valores por geração:")
    with st.expander("Mostrar valores"):
        for geracao, fitness in valores_fitness_geracao:
            st.write(f"Geração {geracao}: {fitness}")

    # Combinando dados em um DataFrame para o gráfico
    dados_grafico = pd.DataFrame(populacoes)

    # Criando o gráfico com Altair
    grafico = alt.Chart(dados_grafico).mark_point().encode(
        x='x',
        y='y',
        color='geracao:O',
        tooltip=['geracao', 'x', 'y']
    ).properties(
        title='Evolução da População por Geração'
    )

    # Exibindo o gráfico no Streamlit
    st.altair_chart(grafico, use_container_width=True)

    # Define a função
    def f(x, y):
      return 15 + x * np.cos(2 * x * np.pi) + y * np.cos(14 * y * np.pi)
    
    # Cria uma grade de pontos
    x = np.linspace(-3.1, 12.1, 100)
    y = np.linspace(4.1, 5.8, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calcula os valores da função para cada ponto
    Z = f(X, Y)
    
    # Cria o gráfico 3D
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z)])
    
    # Define os labels dos eixos
    fig.update_layout(
        title="Gráfico da Função f(x,y)",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)"
        )
    )
    
    # Exibe o gráfico
    fig.show()
