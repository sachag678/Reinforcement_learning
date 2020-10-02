import numpy as np
import time

class NN:

    def __init__(self, layers):
        self.weights = []
        for i in range(0, len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]))

    def forward(self, input_):
        for w in self.weights:
            input_ = np.dot(input_, w)
            input_[input_<0] = 0
        return input_



if __name__ == "__main__":
    layer_size = [2, 50, 1]
    nn = NN(layer_size)

    input_val = np.random.randn(1,2)

    res = nn.forward(input_val)
    print(res)



