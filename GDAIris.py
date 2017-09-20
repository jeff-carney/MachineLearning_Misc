import pandas as pd 
import numpy as np
import feather
import matplotlib.pyplot as plt

# df = pd.read_csv('iris.csv', index_col = 0)

# feather.write_dataframe(df, 'iris.feather')

df = feather.read_dataframe('iris.feather')

x = np.array(df[['Sepal.Length', 'Petal.Width']])
y = np.array(df['Species'])


def discrim_analysis(x, y, lamb, linked = False):
	labels = np.unique(y)
	pi = {}
	mu = {}
	cov = {}

	for label in labels:
		pi[label] = (y == label).sum()/y.shape[0]
		mu[label] = x[y==label].mean(axis = 0)
		cov[label] = (x[y==label] - mu[label]).T @ (x[y==label] - mu[label]) / (y==label).sum()

	if linked:
		new_cov = np.zeros(cov[labels[0]].shape[0])

		for label in labels:
			new_cov = new_cov + ((y==label).sum() * cov[label])

		new_cov = new_cov / y.shape[0]
		new_cov = lamb*(np.eye(new_cov.shape[0]) * np.diag(new_cov)) + ( (1 - lamb) * new_cov )

		for label in labels:
			cov[label] = new_cov

	return pi, mu, cov

def prob_numerator(pi, x, mu, cov):

	difference = (x - mu).reshape(mu.shape[0], 1)
	return ((pi * np.exp((-1/2)*difference.T @ np.linalg.inv(cov) @ difference)) / ( 2* np.pi * np.sqrt( np.linalg.det(cov) ) ))[0,0]


def predict_probs(pi, x, mu, cov, labels):

	probs = np.zeros((x.shape[0], len(pi) ))

	for i in range(x.shape[0]):
		for j in range(len(pi)):
			probs[i, j] = prob_numerator(pi[labels[j]], x[i,:], mu[labels[j]], cov[labels[j]])

	predicted = np.argmax(probs, axis = 1)
	preds = [labels[predicted[i]] for i in range(len(predicted))]

	return preds


pi, mu, cov = discrim_analysis(x, y, 0.0)

pred = predict_probs(pi, x, mu, cov, np.unique(y))

accuracy = (y == pred).sum() / len(pred)

print("Linked = False", "LAMB = ", 0.0, "ACCURACY = " , accuracy)


for lamb in np.arange(0.0, 1.05, 0.05):

	pi, mu, cov = discrim_analysis(x, y, lamb, linked = True)

	pred = predict_probs(pi, x, mu, cov, np.unique(y))

	accuracy = (y == pred).sum() / len(pred)

	print("Linked = True", "LAMB = ", lamb, "ACCURACY = ", accuracy)


# optimal model was unlinked
pi, mu, cov = discrim_analysis(x, y, 0.0)

pred = predict_probs(pi, x, mu, cov, np.unique(y))

accuracy = (y == pred).sum() / len(pred)

data_colors = ['go', 'ko', 'wo']
mu_colors = ['yo', 'ro', 'bo']

labels = np.unique(y)

for i in range(len(labels)):
	label = labels[i]
	plt.plot(x[y == label][:,0], x[y == label][:,1], data_colors[i])
	plt.plot(mu[label][0], mu[label][1], mu_colors[i])

plt.show()

