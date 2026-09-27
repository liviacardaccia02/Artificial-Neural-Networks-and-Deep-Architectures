import numpy as np

class SOM:
    def __init__(self, grid_size, input_dim, learning_rate=0.2):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = np.random.rand(*self.grid_size, self.input_dim)

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=-1)
        bmu_index = np.unravel_index(np.argmin(distances), self.grid_size)
        return bmu_index

    def _get_topological_distance(self, bmu_index):
        """
        Calculate the topological distance between the BMU and each node in the grid.
        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def _get_radius(self, epoch, total_epochs):
        """
        Calculate the neighborhood radius for the current epoch.
        Must be implemented in subclasses.
        """
        raise NotImplementedError
    
    def fit(self, X, epochs):
        for epoch in range(epochs):
            radius = self._get_radius(epoch, epochs)

            for x in X:
                bmu_index = self._find_bmu(x)
                topological_distance = self._get_topological_distance(bmu_index)
                in_neighborhood = topological_distance <= radius # Boolean mask of the nodes in the neighborhood
                mask = in_neighborhood[..., np.newaxis]  # Expand the mask so it matches the weights dimensions
                self.weights += mask * self.learning_rate * (x - self.weights)  # Update weights only for nodes in the neighborhood

class AnimalSOM(SOM):
    def _get_topological_distance(self, bmu_index):
        grid_coords = np.indices(self.grid_size)[0]
        return  np.abs(grid_coords - bmu_index[0])

    def _get_radius(self, epoch, total_epochs):
        decay_rate = 49 / (total_epochs - 1)
        radius = 50 - (epoch * decay_rate)
        return max(int(np.round(radius)), 1)  # Ensure radius is at least 1

class TourSOM(SOM):
    def _get_topological_distance(self, bmu_index):
        # Distance on a ring is the shorter of the direct path and wraparound path.
        n_nodes = self.grid_size[0]
        node_indices = np.arange(n_nodes)
        direct_distance = np.abs(node_indices - bmu_index[0])
        return np.minimum(direct_distance, n_nodes - direct_distance)

    def _get_radius(self, epoch, total_epochs):
        # Use a smaller neighborhood for the smaller tour map: 2, then 1,
        # and finally 0 as training progresses.
        if total_epochs <= 1:
            return 0
        return max(0, int(np.round(2 - 2 * epoch / (total_epochs - 1))))

class VotesSOM(SOM):
    def _get_topological_distance(self, bmu_index):
        # MANHATTAN DISTANCE: |row - bmu_row| + |col - bmu_col|
        grid_coords = np.indices(self.grid_size)
        row_diff = np.abs(grid_coords[0] - bmu_index[0])
        col_diff = np.abs(grid_coords[1] - bmu_index[1])
        return row_diff + col_diff

    def _get_radius(self, epoch, total_epochs):
        # Decay from 5 down to 0
        decay_rate = 5 / (total_epochs - 1)
        radius = 5 - (epoch * decay_rate)
        return max(int(np.round(radius)), 0)