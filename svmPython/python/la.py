from svmutil import *

y, x = svm_read_problem('../LINEAL_SEPARABLES_SVM')
salidas=y[0:100]+y[144:244]
entradas=x[0:100]+x[144:244]
test=x[100:143]+x[244:292]
zeros = [0] * len(test)
m = svm_train(y, x, '-t 0 -s 1')
p_label, p_acc, p_val = svm_predict(zeros, test, m)

support_vector_coefficients = m.get_sv_coef()
support_vectors = m.get_SV() 

#print p_label
print support_vectors
#print support_vector_coefficients