import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt

#Classe Rede Neural
class NeuralODE():
    def __init__(self):
        #npr.RandomState(42)
        self.first_layer={'W':0.1*npr.rand(1,5),'b':0.1*npr.rand(5)}
        self.second_layer={'W':0.1*npr.rand(5,5),'b':0.1*npr.rand(5)}
        self.third_layer={'W':0.1*npr.rand(5,1),'b':0.1*npr.rand(1)}            

    def updateWeights(self,weights):
        self.first_layer['W']=weights[0][0]
        self.first_layer['b']=weights[0][1]

        self.second_layer['W']=weights[1][0]
        self.second_layer['b']=weights[1][1]

        self.third_layer['W']=weights[2][0]
        self.third_layer['b']=weights[2][1]

    def getMyWeights(self):
        weights=[]
        weights.append([self.first_layer['W'],self.first_layer['b']])
        weights.append([self.second_layer['W'],self.second_layer['b']])
        weights.append([self.third_layer['W'],self.third_layer['b']])
        return weights

def initial_weigth(scale, layer_sizes, rs=npr.RandomState(42)):
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


# Criando a Rede Neural
neural_network=NeuralODE()
weights = initial_weigth(0.1, layer_sizes=[1, 5,5, 1])

#FeedFoward da Rede Neural
def y(weights, inputs):
    for W, b in weights:
        outputs = np.dot(inputs, W) + b # Processamento
        inputs = np.tanh(outputs)  # Função de Ativação  
    return outputs

#Derivadas da saída da Rede Neural
ẏ = egrad(y, 1) #Calculando a primeira derivada
ÿ = egrad(ẏ, 1) #Calculando a segunda derivada


# Parâmetros a serem otimizados pela Rede Neural.
# Tanto os pesos de estimação como a energia do sistema
# devem ser estimados.
params = {'w': weights, 'E':1.5}


#####################################################
#                                                   #
#        Vamos resolver o seguinte problema:        #
#                                                   #
#            -0.5*ÿ(x)+V(x)y(x)=E*y(x)              #
#                                                   #
#           Com y(0)=0, y(2)=0 e V(x)=0             #
#                                                   #
#####################################################


# Configuração inicial para o sistema.
# a,b : pontos extremos (condição de contorno)
a=0   
b=2


# Definindo os Pontos da Malha
x = np.linspace(a, b, 200)[:, None]
Δx=x[1]-x[0]


#Definindo o potencial. Nesse exemplo, constante e igual a zero.
def potential(x):
    return 0

V=potential(x)

# Definindo a função de callback
def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteração {0:4d} / Função Custo: {1}".format(step,
                                                      loss_function(params, step)))

# Definindo espaços para armazenar variáveis a serem plotadas
pred=[]
steps = []
loss_by_time=[]


def loss_function(params, step):
    #Obtendo os pesos da Rede Neural e a Energia
    weights = params['w']
    E = params['E']
    #Estimando a função      
    y_pred=y(weights, x)
    ÿ_pred=ÿ(weights, x)

    #Calculando o erro em relação a equação de schrodinger
    error = (-0.5*ÿ_pred + (V*y_pred))-(E * y_pred)

    #Condições de Contorno
    left_boundery = y(weights, a)
    right_boundery = y(weights, b)

    #Estimando a probabilidade  
    y2 = y_pred**2
    prob = np.sum((y2[1:] + y2[0:-1])*Δx/2)

    #Adquirindo parâmetros de cada iteração para posterior plot
    if step % 1000 == 0:
        pred.append([y_pred,step]) 

    loss=np.mean(error**2) + left_boundery**2 + right_boundery**2 + (1.0 - prob)**2
    
    loss_str = str(loss)
    loss_by_time.append(float(loss_str[(loss_str.index('[')+2):-2]))
    return loss

#Otimizando a Rede Neural
params = adam(grad(loss_function), params,step_size=0.001, num_iters=7001, callback=callback)

# E=params[1]

# print("O chute inicial para a energia foi de 0.2 Hartree. A energia final obtida foi E={0.4f} Hartree.\n".format(E))


# Plotando o gráfico do custo
fig=plt.figure(figsize=(10, 8))
plt.plot(loss_by_time)
plt.xlabel('Número de Épocas')
plt.ylabel('Loss Function')
plt.title('Função de Custo por época')
plt.show()


#Plotando as aproximações a cada 1000 épocas
fig=plt.figure(figsize=(10, 8))
k=np.sqrt(2*0.2699)
plt.plot(x,pred[1][0],label=str(pred[1][1])+' Épocas')
plt.plot(x,pred[3][0],label=str(pred[3][1])+' Épocas')
plt.plot(x,pred[5][0],label=str(pred[5][1])+' Épocas')
plt.plot(x,pred[7][0],label=str(pred[7][1])+' Épocas')
plt.plot(x,pred[9][0],label=str(pred[9][1])+' Épocas')
plt.plot(x,pred[11][0],label=str(pred[11][1])+' Épocas')
plt.plot(x,pred[13][0],label=str(pred[13][1])+' Épocas')
plt.plot(x,pred[15][0],label=str(pred[15][1])+' Épocas')
plt.plot(x, np.sqrt(2/2)*np.sin(np.pi*x/2), 'r--', label='Solução Analítica')
plt.ylabel(r'$\psi$(x) (em raios de Bohr)')
plt.xlabel(r'x (em raios de Bohr)')
plt.legend()
plt.title('Aproximações a cada época de treino')
plt.show()
