import matplotlib.pyplot as plt
from sklearn import linear_model

import StocksDataUtils as st
from StockModels import *
from MC_inference import *

import numpy as np
from sklearn.linear_model import LinearRegression

y_sim, true_x = SVM1_sintetic_data(0.95, -0.8, 0.3, -0, 1000, burn_in=100)

x_guess = 2*np.log(abs(y_sim))
#print(x_guess)
regr = linear_model.LinearRegression()
regr.fit(x_guess[:-1].reshape((-1,1)), x_guess[1:].reshape((-1,1)))

#print(regr.score(x_guess[:-1], x_guess[1:]))
print("coeff",regr.coef_)
print("intercept",regr.intercept_)

plt.plot(x_guess[:-1],x_guess[1:],'o' )
plt.show()
