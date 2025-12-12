# Reference are:
# An Introduction to Statistical Learning with Application in Python

# Idea imagine X represent TV advertising and Y may represent sales.
# Then we can regress sales onto TV by fitting the model

# Y = beta_0 + beta_1*X
# sales = beta_0 + beta_1*TV

# Univariate Linear Regression for now

class LinearRegression:
    def __init__(self):
        self.beta_0 = None
        self.beta_1 = None

    def fit(self, X, y):
        n = len(X) # to find the number of elements in input X array

        # Calculate the mean
        x_mean = sum(X) / n
        y_mean = sum(y) / n

        # beta_0 code equavalent to the RSS formula on page 80 (Slope)
        numerator = 0
        denominator = 0
        for i in range(n):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2

        self.beta_1 = numerator / denominator

        # beta_1 code equavalent to the RSS formula on page 80 (Intercept)
        self.beta_0 = y_mean - self.beta_1 * x_mean

    def predict(self, X):
        # Equavalent code Y = beta_0 + beta_1 * X
        return [self.beta_0 + self.beta_1 * x for x in X]


    def calculate_rss(self, X, y):
        n = len(X)
        rss = 0
        for i in range(n):
            y_pred = self.beta_0 + self.beta_1 * X[i]
            rss += (y[i] - y_pred) ** 2
        return rss