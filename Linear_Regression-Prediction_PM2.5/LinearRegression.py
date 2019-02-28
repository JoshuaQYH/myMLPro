import numpy as np

class linearRegression(object):
    def __init__(self, features_num = 1, iteration_times = 10000, learn_rate = 1, eps = 0.0001):
        self.features_num = features_num
        self.iteration_times = iteration_times
        self.learn_rate = learn_rate
        self.eps = eps
        self.weight_bias = None

    def fit(self, train_data_X, label_y):
        """
        fit method:
        1. init weight and bias;
        2. calcualte the gradient vector
        3. update the weight and bias with the adaptive gradient
        4. if update time is equal to max time, fit stops.
        """
        # initial weight and bias
        pre_weight_bias = np.arange(0, self.features_num + 1, 1).T
        cur_weight_bias = np.arange(-100, -100 + self.features_num + 1, 1).T


        # fit and iteration process
        grad_square_set = []
        cmp_grad = None
        cmp_nan = None
        count = 1
        print(train_data_X)
        while count < self.iteration_times:
            pre_weight_bias = cur_weight_bias
            param_grad = 2 * (train_data_X).T.dot(train_data_X.dot(cur_weight_bias) - label_y)
            cur_weight_bias = cur_weight_bias - self.learn_rate  * param_grad
           
            #print(param_grad)
            count += 1
        self.weight_bias = cur_weight_bias
        return True

    def predict(self, test_data):
        return self.weight_bias.dot(test_data)

    def score(self, test_data_X, label_y):
        right = 0
        for i in range(test_data_X.ndim):
            if label_y[i] == self.predict(test_data_X[i]):
                right += 1
        return right * 1.0 / test_data_X.ndim
