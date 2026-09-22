import numpy as np

class SOM:
    def __init__(self, grid_size, input_dim, learning_rate=0.1, sigma=None):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(grid_size) / 2.0
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)

    def _neighborhood_function(self, distance):
        return np.exp(-distance ** 2 / (2 * self.sigma ** 2))

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def fit(self, X, epochs=100):
        for epoch in range(epochs):
            for x in X:
                bmu_index = self._find_bmu(x)
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                        influence = self._neighborhood_function(distance_to_bmu)
                        self.weights[i, j] += influence * self.learning_rate * (x - self.weights[i, j])