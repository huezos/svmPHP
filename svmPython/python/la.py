#Libreria de svm/libsvm
from svmutil import *

#Cargar el archivo en las listas x y y
y, x = svm_read_problem('../LINEAL_SEPARABLES_SVM')

#Creacion de la lista de salidas con los primeros
#100 de la primera y segunda clase
salidas=y[0:100]+y[144:244]

#Creacion de la lista de entradas con los primeros
#100 de la primera y segunda clase
entradas=x[0:100]+x[144:244]

#Creacion de la lista de prueba con el resto de los datos
test=x[100:143]+x[244:292]

#Creación del vector de zeros
zeros = [0] * len(test)

#Entrenamiento del svm con entrenamiento tipo lineal
modelo = svm_train(y, x, '-t 0 -s 1')

#Prediccion de las clases de la lista test
#e_finales = Etiquetas Finales
#p_acc = accuracity/presicion
#p_val = valores probables
e_finales, p_acc, p_val = svm_predict(zeros, test, modelo)

#Vectores soporte
vS = modelo.get_SV() 

print support_vectors