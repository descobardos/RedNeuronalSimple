import numpy as np 


def Softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp

#Matriz de pesos sinapticos aleatorios
W = np.random.randn(3, 4) * 0.01

#Pixeles generados sinteticos 
x = np.array([[0.1852, 0.0754, 0.437, 0.0619]]).T

#Matriz ONE-HOT asumiento que la clase es perro
y = np.array([[0, 1, 0]]).T

#Matriz de biass
b = np.zeros((3, 1))

#Calculo de salid de Nodo o scores
z = W@x + b

#Calculo de prediccion o probabilidad
y_hat = Softmax(z)
print(y_hat)

#Calculo funcion de perdida para la clase y donde 1 es perro
loss_i = -np.log(y_hat[np.where(y==1)])[0]

dz = y_hat - y
dw = dz@x.T 
db = dz 

learning_rate = 1e-2
W = W - learning_rate*dw
b = b - learning_rate*db

