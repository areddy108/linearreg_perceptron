import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    transformed_features = np.zeros(features.shape)
    for i in range(np.size(features,0)):
            transformed_features[i, 0] = np.sqrt(np.square(features[i,0]) + np.square(features[i,1]))
            transformed_features[i, 1] = np.arctan(features[i,1]/features[i,0])
    return(transformed_features)
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    #raise NotImplementedError()

class Perceptron():
    def __init__(self, max_iterations=200, weights = None):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.weights = weights
        #raise NotImplementedError()

    def target(self, weights, features, targets):

        for i in range(np.size(features, 0)):
            if np.dot(weights, features[i, :]) *  targets[i] <= 0:
                return True

        return False

    def H(self, int):
        if(int > 0):
            return 1
        else:
            return -1

    def fit(self, features, targets):

        weights = np.array([1, 2, 3])
        oneF = np.ones((np.size(features,0), 1))
        features = np.column_stack((oneF, features))
        count = 0
        while(count < self.max_iterations and self.target(weights, features, targets)):
            count = count + 1
            for i in range(np.size(features, 0)):

                if(targets[i] * (weights[0]*1 + weights[1]*features[i, 1] + weights[2]*features[i, 2]) <=0):
                    weights = weights + features[i, :]*targets[i]

        self.weights = weights



        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        #raise NotImplementedError()

    def predict(self, features):
        predictions = np.zeros( np.size(features, 0))
        oneF = np.ones((np.size(features,0), 1))
        features = np.column_stack((oneF, features))
        for i in range(np.size(features, 0)):
            predictions[i] = self.H((self.weights[0] * 1 + self.weights[1] * features[i, 1] + self.weights[2] * features[i, 2]))

        return predictions
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        #raise NotImplementedError()

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        #raise NotImplementedError()
