import numpy as np
import scipy.stats as stats


class LinearRegression:
    def __init__(self, y, X):
        self.y = y
        self.X = np.column_stack([np.ones(y.shape[0]), X])

    @property
    def fit(self):  # estimates coefficients
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    @property
    def d(self):  # contains the number of features
        return self.X.shape[1] - 1

    @property
    def n(self):  # contains the size of the sample
        return self.y.shape[0]

    @property
    def SSE(self):  # estimates the amount of variability that can not be explained by the linear model
        return np.sum(np.square(self.y - (self.X @ self.fit)))

    @property
    def variance(self):  # calculates the variance
        return self.SSE/(self.n-self.d-1)

    @property
    def std_deviation(self):  # calculates the standard deviation
        return np.sqrt(self.variance)

    @property
    def Syy(self):  # calculates the totat variability
        return (self.n*np.sum(np.square(self.y)) - np.square(np.sum(self.y)))/self.n

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
    def c(self):
        return np.linalg.pinv(self.X.T @ self.X)*self.variance

    @property
    def Rsq(self):  # estimates the extent to which the model fits the data
        return self.SSR/self.Syy

    @property
    def confidence_level(self):
        return round(self.Rsq, 3)

    @property  # calculates the Pearson correlation between all pairs of parameters
    def individuall_corr(self):
        return [stats.pearsonr(self.X[:, i], self.X[:, j])[:][0]
                for i in range(1, self.d+1) for j in range(i+1, self.d+1) if i != j]

    @property
    def individuall_signif(self):
        return [2*min(stats.t.cdf(self.fit[i]/(self.std_deviation*np.sqrt(self.c[i, i])), self.n-self.d-1),
                      stats.t.sf(self.fit[i]/(self.std_deviation*np.sqrt(self.c[i, i])), self.n-self.d-1))
                for i in range(1, self.d+1)]

    @property
    def print_p_value(self):
        features = ['Kinematic', 'Geometric', 'Inertial', 'Observer']

        for f, val in zip(features, self.individuall_signif):
            print(f"P-value for {f}: {val}")
