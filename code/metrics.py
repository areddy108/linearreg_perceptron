import numpy as np

def mean_squared_error(estimates, targets):

    MSE = np.sum(np.square(np.subtract(estimates, targets)))/estimates.size
    return MSE
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is 
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy.

    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    #raise NotImplementedError()