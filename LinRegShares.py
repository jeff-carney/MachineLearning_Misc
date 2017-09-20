import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(1)

df = pd.read_csv('online_news_popularity.csv')
df = shuffle(df)

train = df.iloc[0:int((2.0/3.0)*len(df)),:]
validation = df.iloc[int((2.0/3.0)*len(df)):int((5.0/6.0)*len(df)),:]
test = df.iloc[int((5.0/6.0)*len(df)):,:]

x_train = train[[col for col in df.columns if col not in ['url', ' shares']]]
y_train = np.log(train.loc[:,' shares']).reshape(-1,1)

x_val = validation[[col for col in df.columns if col not in ['url', ' shares']]]
y_val = np.log(validation.loc[:,' shares']).reshape(-1,1)

x_test = test[[col for col in df.columns if col not in ['url', ' shares']]]
y_test = np.log(test.loc[:,' shares']).reshape(-1,1)

x_train = np.hstack(( np.ones_like(y_train) , x_train ))
x_val = np.hstack(( np.ones_like(y_val) , x_val ))
x_test = np.hstack(( np.ones_like(y_test) , x_test ))

def linreg(x, y, lamb):

	eye = np.eye(x.shape[1])
	eye[0,0] = 0.0
	w = np.linalg.solve( (x.T @ x) + (lamb*eye), x.T @ y )

	return w



def linreg_cv(x, y, xVal, yVal, minLamb, maxLamb, intvl):

	lambdaList = np.arange(minLamb, maxLamb, intvl).tolist()

	w = linreg(x, y, minLamb)
	minRMSE = np.sqrt(((1.0/len(xVal))*(np.inner((yVal - xVal @ w).T, (yVal - xVal @ w).T))))
	optLamb = minLamb

	rmseList = []
	weightNorms = []

	for lamb in lambdaList:

		w = linreg(x, y, lamb)
		rmse = np.sqrt((1.0/len(xVal))*((yVal - xVal @ w).T @ (yVal - xVal @ w)))
		print(lamb)
		rmseList.append(rmse[0,0])
		weightNorms.append(np.sqrt(w.T @ w)[0,0])

		if rmse < minRMSE:
			minRMSE = rmse
			optLamb = lamb
	
	return optLamb, rmseList, lambdaList, weightNorms

optLamb, rmseList, lambdaList, weightNorms = linreg_cv(x_train, y_train, x_val, y_val, 0.0, 150.0, 0.01)

print(optLamb)

plt.plot(lambdaList, rmseList)
plt.xlabel("Lambda")
plt.ylabel("Cost")
plt.xlim([-10,160])
plt.show()

plot = plt.plot(lambdaList, weightNorms)
plt.xlabel("Lambda")
plt.ylabel("Weight Norm")
plt.ylim([6,8])
plt.xlim([-10,160])
plt.show()