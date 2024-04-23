import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable

print("\x1b[2J\x1b[1;1H") 

# sen(X)*sen(2X).
entradas = 1
#neur = 200 
#alfa = 0.005
#errotolerado = 0.02 # Testes: 0.5 / 0.1 / 0.05 / 0.02.
listaciclo = []
listaerro = []
xmin = -1 # Limite inferior da funo.
xmax = 1 # Limite superior da funo.
npontos = 50 # Nmero de pontos igualmente espaados.

# Lista de valores para cada parâmetro variado
alfas = [0.001, 0.005, 0.01]  # Diferentes taxas de aprendizagem
erros_tolerados = [0.05, 0.02, 0.01]  # Diferentes valores de erro tolerado
faixa_aleatoria = [1, 1.1, 1.1]  # Diferentes faixas de valores para inicialização dos pesos
faixa_aleatoria2 = [0.1, 0.2, 0.3]  # Diferentes faixas de valores para inicialização dos pesos
ciclos_maximos = [500, 1000, 1500]  # Diferentes valores máximos de ciclos
neurs = [50, 100, 150 , 200]

#Gerando o arquivo de entradas
x_orig = np.linspace(xmin,xmax,npontos) # Criao dos pontos igualmente espaados.
x = np.zeros((npontos,1))
for i in range(npontos):
    x[i][0]=x_orig[i] # Entradas prontas para a RNA.
#End For

(amostras,vsai) = np.shape(x) # 50 amostras e 1 sada.

t_orig = (np.sin(x))*(np.sin(2*x)) # Target "puro".
t = np.zeros((1,amostras))
for i in range(amostras):
    t[0][i]=t_orig[i] # Target pronto para a RNA.
#End For

(vsai,amostras) = np.shape(t) # [50][1]

resultados = []


for alfa in alfas:
    for errotolerado in erros_tolerados:
        for aleatorio1 in faixa_aleatoria:
            for aleatorio2 in faixa_aleatoria2:
                for ciclo_maximo in ciclos_maximos:
                    for neur in neurs:
                    
                        # Gerando os pesos sinpticos aleatoriamente.
                        v = np.zeros((entradas,neur))
                        #aleatorio = 1
                        for i in range(entradas):
                            for j in range(neur):
                                v[i][j] = rd.uniform(-aleatorio1,aleatorio1)
                            #End For
                        #End For
                        
                        v0 = np.zeros((1,neur))
                        for j in range(neur):
                            v0[0][j] = rd.uniform(-aleatorio1,aleatorio1)
                        #End For
                        
                        w = np.zeros((neur,vsai))
                        #aleatorio = 0.2
                        for i in range(neur):
                            for j in range(vsai):
                                w[i][j] = rd.uniform(-aleatorio2,aleatorio2)
                            #End For
                        #End For
                        
                        w0=np.zeros((1,vsai))
                        for j in range(vsai):
                            w0[0][j]=rd.uniform(-aleatorio2,aleatorio2)
                        #End For
                        
                        # Matrizes de atualizao de pesos e valores de saida da rede.
                        vnovo = np.zeros((entradas,neur))
                        v0novo = np.zeros((1,neur))
                        wnovo = np.zeros((neur,vsai))
                        w0novo = np.zeros((1,vsai))
                        zin_j = np.zeros((1,neur))
                        z_j = np.zeros((1,neur))
                        deltinha_k = np.zeros((vsai,1))
                        deltaw0 = np.zeros((vsai,1))
                        deltinha_j = np.zeros((1,neur))
                        x_linhaTransp = np.zeros((1,entradas))
                        y_transp = np.zeros((vsai,1))
                        t_transp = np.zeros((vsai,1))
                        deltinha_jTransp = np.zeros((neur,1))
                        ciclo = 0
                        errototal=1
                        
                        while errotolerado < errototal and ciclo < ciclo_maximo:
                            errototal=0
                            for padrao in range(amostras):
                                for j in range(neur):
                                    zin_j[0][j] = np.dot(x[padrao,:],v[:,j]) + v0[0][j]
                                #End For
                         
                                z_j = np.tanh(zin_j) # Funo de ativao.
                                
                                yin = np.dot(z_j,w) + w0 # Saída pura.
                                y = np.tanh(yin) # Sada lquida.
                             
                                for m in range(vsai):
                                    y_transp[m][0] = y[0][m] 
                                #End For
                                
                                for m in range(vsai):
                                    t_transp[m][0]=t[0][padrao]
                                #End For
                                errototal = errototal+(0.5*(np.sum(((t_transp-y_transp)**2))))
                                
                                # Busca das matrizes para atualizao dos pesos.
                                deltinha_k = (t_transp - y_transp)*(1 + y_transp)*(1 - y_transp)
                                deltaw = alfa*(np.dot(deltinha_k,z_j))
                                deltaw0 = alfa * deltinha_k
                                deltinhain_j = np.dot(np.transpose(deltinha_k),np.transpose(w))
                                deltinha_j = deltinhain_j * (1 + z_j) * (1 - z_j)
                                for m in range(neur):
                                    deltinha_jTransp[m][0] = deltinha_j[0][m]
                                #End For
                                for k in range(entradas):
                                    x_linhaTransp[0][k] = x[padrao][k]
                                #End For
                                
                                deltav = alfa * np.dot(deltinha_jTransp,x_linhaTransp)
                                deltav0 = alfa * deltinha_j
                                
                                #Realizando as atualizaes de pesos e bias.
                                vnovo = v + np.transpose(deltav)
                                v0novo = v0 + np.transpose(deltav0)
                                
                                wnovo = w + np.transpose(deltaw)
                                w0novo = w0 + np.transpose(deltaw0)
                                
                                # Preparo para o prximo lao.
                                v =vnovo
                                v0 = v0novo
                                w = wnovo
                                w0 = w0novo
                            #End: for padrao in range(amostras):   
                            ciclo = ciclo+1
                            listaciclo.append(ciclo)
                            listaerro.append(errototal)
                            print('Ciclo\t Erro')
                            print(ciclo,'\t',errototal)
                        
                            # Comparao target e y.
                            zin2_j = np.zeros((1,neur))
                            z2_j = np.zeros((1,neur))
                            t_teste = np.zeros((amostras,1))
                            
                            for i in range(amostras):
                                for j in range(neur):
                                    zin2_j[0][j] = np.dot(x[i,:],v[:,j])+v0[0][j]
                                    z2_j = np.tanh(zin2_j)
                                #End For
                                yin2 = np.dot(z2_j,w) + w0
                                y2 = np.tanh(yin2)
                                t_teste[i][0] = y2
                            #End For   
                        #End While
                        plt.plot(x,t_orig,color='red')
                        plt.plot(x,t_teste,color='blue')
                        #plt.title('Gráfico do ciclo: ', ciclo)
                        plt.show() 
                        
                        # Calcular RMSE entre as previsões da rede neural (t_teste) e os valores reais (t_orig)
                        rmse = np.sqrt(mean_squared_error(t_orig, t_teste))
                        
                        resultado = {
                            "alfa": alfa,
                            "erro_tolerado": errotolerado,
                            "aleatoriov": aleatorio1,
                            "aleatoriow": aleatorio2,
                            "ciclo_maximo": ciclo_maximo,
                            "neur": neur,
                            "rmse": rmse
                        }
                        resultados.append(resultado)
                    

# https://geo-python.github.io/2017/lessons/L7/matplotlib.html
# Cria uma tabela PrettyTable
tabela = PrettyTable()

# Define os nomes das colunas
tabela.field_names = ["Alfa", "Erro Tolerado", "Aleatório", "Ciclo Máximo", "Neurônios", "RMSE"]

# Adiciona as linhas à tabela com os resultados
for resultado in resultados:
    tabela.add_row([
        resultado["alfa"],
        resultado["erro_tolerado"],
        resultado["aleatorio"],
        resultado["ciclo_maximo"],
        resultado["neur"],
        resultado["rmse"]
    ])

# Imprime a tabela
print(tabela)

# Encontra o menor RMSE
menor_rmse = float('inf')
melhor_resultado = None

for resultado in resultados:
    if resultado["rmse"] < menor_rmse:
        menor_rmse = resultado["rmse"]
        melhor_resultado = resultado

# Imprime o resultado com o menor RMSE
print("Menor RMSE:")
print("Alfa:", melhor_resultado["alfa"])
print("Erro Tolerado:", melhor_resultado["erro_tolerado"])
print("Aleatório:", melhor_resultado["aleatorio"])
print("Ciclo Máximo:", melhor_resultado["ciclo_maximo"])
print("Neurônios:", melhor_resultado["neur"])
print("RMSE:", melhor_resultado["rmse"])

# Gráfico com as variáveis utilizadas no menor RMSE
alfa_menor_rmse = melhor_resultado["alfa"]
erro_tolerado_menor_rmse = melhor_resultado["erro_tolerado"]
aleatorio_menor_rmse = melhor_resultado["aleatorio"]
ciclo_maximo_menor_rmse = melhor_resultado["ciclo_maximo"]
neur_menor_rmse = melhor_resultado["neur"]

plt.plot(x, t_orig, color='red', label='Target')
plt.plot(x, t_teste, color='blue', label='Previsão')
plt.title('Gráfico com as variáveis do menor RMSE\nAlfa: {} Erro Tolerado: {} Aleatório: {} Ciclo Máximo: {} Neurônios: {}'.format(
    alfa_menor_rmse, erro_tolerado_menor_rmse, aleatorio_menor_rmse, ciclo_maximo_menor_rmse, neur_menor_rmse))
plt.legend()
plt.show()
