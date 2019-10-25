import numpy as np
import time

class NN:

    def __init__(self, layers):
        self.weights = []
        for i in range(0, len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]))

    def forward(self, input_):
        values = input_
        for w in self.weights:
            values = np.dot(values, w)
        return values



if __name__ == "__main__":
    layer_size = [2, 8, 1]
    nn = NN(layer_size)

    start = time.time()
    for i in range(100000):
        nn.forward(np.array([2, 1]))
    end = time.time()
    print(end - start)



