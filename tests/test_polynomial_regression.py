import numpy as np
from code import generate_regression_data, PolynomialRegression, mean_squared_error

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def test_polynomial_regression():
    degrees = range(10)
    amounts = [10, 100, 1000, 10000]

    

    for degree in degrees:
        p = PolynomialRegression(degree)
        for amount in amounts:
            x, y = generate_regression_data(degree, amount, amount_of_noise=0.0)
            p.fit(x, y)
            y_hat = p.predict(x)
            mse = mean_squared_error(y, y_hat)
            assert (mse < 1e-1)
    return

    #plt.scatter(np.arrange(10), mseTest).show


