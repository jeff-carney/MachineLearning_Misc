import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mnist_train.csv', header = None)
df_test = pd.read_csv('mnist_test.csv', header = None)


x_train = df.iloc[:,1:]/255
y_train = df.iloc[:,0]

x_test = df_test.iloc[:,1:]/255
y_test = df_test.iloc[:,0]

def softmax(x):

	# NOTE: this is the same as returning np.exp(x)/np.sum(np.exp(x),axis = 1) but the max is subtracted off for numerical stability

	s = np.exp(x - np.max(x, axis = 1).reshape(x.shape[0], 1))
	return s/np.sum(s, axis = 1).reshape(s.shape[0],1)

def oneHotEncode(y, num_classes):

	s = np.zeros((y.shape[0], num_classes))

	for i in range(y.shape[0]):
		s[i, y.iloc[i]] = 1
	
	return s

def log_likelihood(w, x, y_oneHot_bool, lamb = 1e-6):

	mu = softmax(x @ w)
	return np.sum( mu[y_oneHot_bool] - np.log(np.sum(np.exp( mu ))) ) + lamb*np.trace(w.T @ w)


def gradient(w, x, y_oneHot_bool, lamb = 1e-6):

	mu = softmax( x @ w )

	return x.T @ (mu - y_oneHot_bool) + lamb*w

def softmax_grad_descent(x, y, num_classes,lamb = 1e-6, grad_check = 1e-2, max_iter = 500, step_size = 2.2e-5):
	print("Start", lamb)
	w = np.zeros((x.shape[1], num_classes))

	y_oneHot_bool = oneHotEncode(y_train, num_classes).astype(bool)

	objFuncList = [log_likelihood(w, x, y_oneHot_bool, lamb = lamb)]

	grad = gradient(w, x, y_oneHot_bool, lamb = lamb)

	while(len(objFuncList) < max_iter and np.linalg.norm(grad) > grad_check):

		w = w - grad*step_size
		objFunc = log_likelihood(w, x, y_oneHot_bool, lamb = lamb)
		
		objFuncList.append(objFunc)

		grad = gradient(w, x, y_oneHot_bool, lamb = lamb)

	print("End", lamb)
	return w, objFuncList


lambList = [1e-2, 1e-1, 1, 10, 100, 1000]

best_acc = 0
for lamb_iter in lambList:
	w, objFuncList = softmax_grad_descent(x_train, y_train, 10, lamb = lamb_iter, max_iter = 500)

	mu = softmax(x_test @ w)

	y_pred = np.argmax(mu, axis = 1)

	accuracy = (np.sum((y_pred == y_test).astype(int))/y_pred.shape[0])
	print(accuracy)
	if accuracy > best_acc:
		best_acc = accuracy
		optW = w
		best_objFuncList = objFuncList
		optLamb = lamb_iter


print(optLamb)
mu = softmax(x_test @ optW)

print(mu)

y_pred = np.argmax(mu, axis = 1)

accuracy = (np.sum((y_pred == y_test).astype(int))/y_pred.shape[0])


# x-vector for plotting gradient performance
x = list(range(1, len(best_objFuncList) + 1))

plt.plot(x, best_objFuncList)
plt.xlabel('Iteration')
plt.ylabel('Objective (J(w))')
plt.xlim([-10,600])
plt.show()



