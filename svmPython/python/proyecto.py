from svmutil import *

#Entrenamiento con 100 datos
y, x = svm_read_problem('LINEAL_SEPARABLES_SVM')

posClase1 = 0
posClase2 = 144

def crearVectoresSoporte(cantidad):
	medio = cantidad/2
	salidaM =  y[posClase1:posClase1+medio] + y[posClase2:posClase2+medio]
	entradaM = x[posClase1:posClase1+medio] + x[posClase2:posClase2+medio]

	entradaT = x[posClase1+medio:posClase2] + x[posClase2+medio:len(x)]
	salidaT  = y[posClase1+medio:posClase2] + y[posClase2+medio:len(x)]

	m = svm_train(salidaM, entradaM, '-t 0 -s 1')
	p_label, p_acc, p_val = svm_predict(salidaT, entradaT, m)

	support_vector_coefficients = m.get_sv_coef()
	support_vectors = m.get_SV() 

	return support_vectors

def crearVectorSoporteConVectoresSoporte(support_vectors):
	salidas = []
	entradas = []
	test = x[:]
	test2 = y[:]

	for vector in support_vectors:
		for j in range(len(test)):
			if (vector[1] == test[j][1] and vector[2] == test[j][2]):
				entradas.append(test.pop(j))
				salidas.append(test2.pop(j))
				break

	zeros = [0] * len(test)
	m = svm_train(salidas, entradas, '-t 0 -s 1')
	p_label4, p_acc4, p_val4 = svm_predict(test2, test, m)

	support_vector_coefficients4 = m.get_sv_coef()
	support_vectors4 = m.get_SV() 

	return support_vectors4

#Reaccion en cadena de los entrenamientos hasta que los vectores se estabilicen
print "|Clasificando con 100|"
support_vectors = crearVectoresSoporte(100)
fin = False
iter = 0

while (not fin and iter < 100):
	print "|Corrida " + str(iter) + "|Entrenando con: " + str(len(support_vectors)) + "|"
	vector_inicial = support_vectors[:]
	support_vectors = crearVectorSoporteConVectoresSoporte(vector_inicial);	

	if (vector_inicial == support_vectors):
		fin = True

	iter += 1

