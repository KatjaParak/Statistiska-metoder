import numpy as np
import scipy.stats as stats


class LinearRegression:
    def __init__(self, Y, X):
        self.Y = Y
        self.X = np.column_stack([np.ones(Y.shape[0]), X])

    @property
    def fit(self):  # estimates coefficients
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y

    @property
    def d(self):  # contains the number of features
        return self.X.shape[1] - 1

    @property
    def n(self):  # contains the size of the sample
        return self.Y.shape[0]

    @property
    def SSE(self):  # estimates the amount of variability that can not be explained by the linear model
        return np.sum(np.square(self.Y - (self.X @ self.fit)))

    @property
    def variance(self):  # calculates the variance
        return self.SSE/(self.n-self.d-1)

    @property
    def std_deviation(self):  # calculates the standard deviation
        return np.sqrt(self.variance)

    @property
    def Syy(self):  # calculates the totat variability
        return (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n

    @property
    def SSR(self):  # calculates the sum of squares due to regression
        return self.Syy - self.SSE

    @property
    def significance_stat(self):
        return (self.SSR/self.d)/self.std_deviation

    @property
    def p_value(self):
        return stats.f.sf(self.significance_stat, self.d, self.n-self.d-1)

    @property
    def significance_param(self):
        return np.linalg.pinv(self.X.T @ self.X)*self.variance

    @property
    def Rsq(self):  # estimates the extent to which the model fits the data
        return self.SSR/self.Syy
