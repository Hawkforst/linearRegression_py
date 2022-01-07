import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# write your code here
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

regSci = LinearRegression(fit_intercept=True)

class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0
        self.R2 = 0
        self.rmseVal = 0
        if fit_intercept:
            self.X = pd.read_csv('data_stage4.csv',sep=',').iloc[:, 0:-1]
            self.Y = pd.read_csv('data_stage4.csv',sep=',').iloc[:, -1]
            self.X.insert(loc=0, column='Intercept col', value=np.ones(self.Y.values.shape[0]))
        else:
            self.X= pd.read_csv('data_stage4.csv',sep=',').iloc[:, 0:-1]
            self.Y = pd.read_csv('data_stage4.csv', sep=',').iloc[:, -1]
        self.y_hat = []

    def fit(self):
        if self.fit_intercept:
            [self.intercept, *self.coefficient] = \
                (np.linalg.inv(np.transpose(self.X.values).dot(self.X.values))).dot(np.transpose(self.X.values)) \
                    .dot(self.Y.values)
        else:
            self.coefficient = \
                (np.linalg.inv(np.transpose(self.X.values).dot(self.X.values))).dot(np.transpose(self.X.values)) \
                    .dot(self.Y.values)

    def predict(self):
        self.y_hat = (self.X.values).dot(np.append(self.intercept,self.coefficient))

    def r2_score(self):
        topSum = 0
        botSum = 0
        y_mean = np.mean(self.Y.values)
        for idx in range(len(self.y_hat)):
            topSum += (self.Y.values[idx] - self.y_hat[idx])**2
            botSum += (self.Y.values[idx] - y_mean)**2
        self.R2 = 1 - topSum / botSum

    def rmse(self):
        sum = 0
        for idx in range(len(self.y_hat)):
            sum += (self.Y.values[idx] - self.y_hat[idx]) ** 2
        sum /= len(self.y_hat)
        self.rmseVal = np.sqrt(sum)

    def results(self):
        return \
         {'Intercept': self.intercept,
         'Coefficient': np.array(self.coefficient),
         'R2': self.R2,
         'RMSE': self.rmseVal}

    def plot(self):
        plt.plot(self.X.values, self.Y)
        plt.show()

regr = CustomLinearRegression()

regr.fit()

regr.predict()

regr.r2_score()

regr.rmse()

fit_results = regr.results()



reg = LinearRegression().fit(regr.X, regr.Y)

sklearn_pred = reg.predict(regr.X)

R2sk = r2_score(y_pred=sklearn_pred,y_true=regr.Y.values)
rmsesk = np.sqrt(mean_squared_error(y_pred=sklearn_pred,y_true=regr.Y.values))

print({'Intercept': reg.intercept_ - fit_results['Intercept'],
         'Coefficient': np.array(reg.coef_)[1:] - np.array(fit_results['Coefficient']),
         'R2': R2sk - fit_results['R2'],
         'RMSE': rmsesk - fit_results['RMSE']})

