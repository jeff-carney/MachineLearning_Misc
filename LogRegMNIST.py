import pandas as pd
import numpy as np
from scipy import sparse, linalg
import matplotlib.pyplot as plt

df = pd.read_csv('mnist_train.csv', header = None)
df_test = pd.read_csv('mnist_test.csv', header = None)

df = df[df.loc[:,0] <= 1]
df_test = df_test[df_test.loc[:,0] <= 1]

print(df)

x_train = df.iloc[:,1:]
x_train = x_train/255
y_train = df.iloc[:,0].reshape(-1,1)

x_test = df_test.iloc[:,1:]
x_test = x_test/255
y_test = df_test.iloc[:,0].reshape(-1,1)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# NLL(w) in murphy
# y_bool is a binary vector converted to a boolean vetor
def log_likelihood(w, x, y_bool, lamb = 1e-6):

	mu = sigmoid(x @ w)

	mu[~y_bool] = 1 - mu[~y_bool]

	return np.log(mu).sum() - (lamb/2)*np.inner(w.T,w.T)

# gradient of the log likelihood function
def grad_log_likelihood(w, x, y, lamb = 1e-6):

	return x.T @ (sigmoid(x @ w) - y) + lamb*w

def newton_step(w, x, y, lamb = 1e-6):

	mu = sigmoid(x @ w)

	return linalg.cho_solve( 
		linalg.cho_factor( x.T @ sparse.diags((mu * (1 - mu)).reshape(len(mu),)) @ x + lamb*sparse.eye(x.shape[1]), ) , 
		grad_log_likelihood(w, x, y, lamb = lamb) ,
		)


# function to solve for optimal parameter vector for logistic regression using gradient descent
def lr_grad(x, y, lamb = 1e-5, max_iters =500, grad_check = 1e-2, step_size = 61e-4):

	# converting our binary y vector to a boolean vector for use in the log_likelihood function to compute the objective
	y = y.astype(bool)

	# initializing our feature vector to a vector of zeros with length equal to the number of columns in our x matrix
	w = np.zeros((x.shape[1],1))

	# initializing a list that will contain all values computed for our objective function
	objFuncList = [log_likelihood(w, x, y, lamb = lamb)[0,0]]

	# initializing gradient of our objective function
	grad = grad_log_likelihood(w, x, y, lamb = lamb)

	# want to run this loop until we have reached the maximum number of iterations or we have reached an optimal paramter vector
	while len(objFuncList) - 1 <= max_iters and np.linalg.norm(grad) > grad_check:

		# computing the gradient
		grad = grad_log_likelihood(w, x, y, lamb = lamb)

		# print(np.linalg.norm(grad))

		# updating the vector of parameters
		w = w - step_size * grad

		j = log_likelihood(w, x, y, lamb = lamb)
		# adding the objective for this iteration with updated parameter vector
		objFuncList.append(j[0,0])

	return w, objFuncList

# function to solve for optimal parameter vector for logistic regression using newton's method
def lr_newton(x, y, lamb = 1e-5, max_iters =500, step_check = 1e-2):

	# converting our binary y vector to a boolean vector for use in the log_likelihood function to compute the objective
	y = y.astype(bool)

	# initializing our feature vector to a vector of zeros with length equal to the number of columns in our x matrix
	w = np.zeros((x.shape[1],1))

	# initializing a list that will contain all values computed for our objective function
	objFuncList = [log_likelihood(w, x, y, lamb = lamb)[0,0]]

	# initializing the newtonian step
	step = newton_step(w, x, y, lamb = lamb)

	# want to run this loop until we have reached the maximum number of iterations or we have reached an optimal paramter vector
	while len(objFuncList) - 1 <= max_iters and np.linalg.norm(step) > step_check:

		# computing the step
		step = newton_step(w, x, y, lamb = lamb)

		# print(np.linalg.norm(step))

		# updating the vector of parameters
		w = w - step

		j = log_likelihood(w, x, y, lamb = lamb)
		# adding the objective for this iteration with updated parameter vector
		objFuncList.append(j[0,0])

	return w, objFuncList

# finding optimal parameter fector using gradient descent
optW, objFuncList = lr_grad(x_train, y_train, lamb = 2, step_size = 5e-4)

# finding optimal parameter fector using newton's method
newtW, newtObjList = lr_newton(x_train, y_train, lamb = 2)

# x-vector for plotting newton performance
newtX = list(range(1, len(newtObjList) + 1))

# x-vector for plotting gradient performance
x = list(range(1, len(objFuncList) + 1))

plt.plot(x, objFuncList)
plt.plot(newtX, newtObjList)
plt.xlabel('Iteration')
plt.ylabel('Objective (J(w))')
plt.xlim([-10,600])
plt.show()

# predicting y values and computing accuracy on our test set with optimal paramters from gradient descent
y_pred_grad = sigmoid(x_test @ optW)
grad_accuracy = (1-(np.abs(y_pred_grad - y_test).sum()/len(y_test)))*100
print("Gradient Descent Accuracy: ",grad_accuracy,"%")

# predicting y values and computing accuracy on our test set with optimal paramters from Newton's method
y_pred_newt = sigmoid(x_test @ newtW)
newt_accuracy = (1-(np.abs(y_pred_newt - y_test).sum()/len(y_test)))*100
print("Newton's Method Accuracy: ",newt_accuracy,"%")