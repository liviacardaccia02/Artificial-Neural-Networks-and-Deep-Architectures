import numpy as np

class RBF:
    def __init__(self, centers, widths, weights):
        self.centers = np.array(centers)
        self.widths = np.array(widths)
        self.weights = np.array(weights)

    def rbf(self, x, center, width):
        return np.exp(-((x - center) ** 2) / (2 * width ** 2))

    def forward(self, X):
        outputs = []
        # Calculate activation of every RBF unit
        for x in X:
            activations = np.array([
                self.rbf(x, center, width)
                for center, width in zip(self.centers, self.widths)
            ])
            # Weighted sum of RBF activations
            output = np.dot(self.weights, activations)
            outputs.append(output)

        return outputs
    
    def fit(self, X, Y):
        X = np.asarray(X).flatten()
        Y = np.asarray(Y).flatten()

        activations = []

        for x in X:
            activations.append([
                self.rbf(x, center, width)
                for center, width in zip(self.centers, self.widths)
            ])

        activations = np.array(activations)

        self.weights = np.linalg.pinv(activations).dot(Y)
        
        
class OnlineRBF(RBF):
    def fit_delta(self, X, Y, epochs=50, lr=0.1):
        X = np.asarray(X).flatten()
        Y = np.asarray(Y).flatten()
        
        # Initialize weights randomly
        self.weights = np.random.randn(len(self.centers))
        epoch_errors = []
        
        for epoch in range(epochs):
            # Randomly shuffle data in each epoch
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            
            total_loss = 0
            for idx in indices:
                x_k, y_k = X[idx], Y[idx]
                
                # Calculate phi (activations for single point)
                phi = np.array([self.rbf(x_k, c, w) for c, w in zip(self.centers, self.widths)])
                
                # Predict and calculate instantaneous error
                pred = np.dot(self.weights, phi)
                error = y_k - pred
                
                # Delta rule weight update
                self.weights += lr * error * phi
                total_loss += error ** 2
                
            epoch_errors.append(total_loss / len(X)) # Store MSE for the epoch
            
        return epoch_errors


class RBFWithVectorQuantization(RBF):
    def __init__(self, centers, widths, weights, cl_threshold=0.1):
        super().__init__(centers, widths, weights)
        self.cl_threshold = cl_threshold

    def fit_cl(self, X, epochs=50, lr=0.1):
        X = np.asarray(X).flatten()
        
        for epoch in range(epochs):
            #select random sample from X
            for _ in range(len(X)):
                x_k = np.random.choice(X, replace=False)
                
                #get the closest center to x_k
                distances = np.abs(self.centers - x_k)
                closest_idx = np.argmin(distances)
                
                # update the closest center and its width
                self.centers[closest_idx] += lr * (x_k - self.centers[closest_idx])

class RBFWithVectorQuantization_2:
    def __init__(self, num_centers, num_winners=2):
        """
        num_centers: Number of RBF units.
        num_winners: Number of units updated per step to prevent dead units.
        """
        self.num_centers = num_centers
        self.num_winners = num_winners
        self.centers = None
        self.widths = None
        self.weights = None

    def fit_cl(self, X, epochs=50, lr=0.1):
        # Convert X to at least 2D to handle both 1D (sin wave) and 2D (ballistics) inputs
        X = np.atleast_2d(X)
        
        # Initialize centers by randomly selecting data points from X
        initial_indices = np.random.choice(len(X), self.num_centers, replace=False)
        self.centers = X[initial_indices].astype(float)
        
        for epoch in range(epochs):
            # Shuffle indices to guarantee every point is evaluated exactly once per epoch
            indices = np.random.permutation(len(X))
            
            for idx in indices:
                x_k = X[idx]
                
                # Compute Euclidean distance across the correct axis for N-dimensional support
                distances = np.linalg.norm(self.centers - x_k, axis=1)
                
                # Identify the top 'num_winners' closest centers to prevent dead units
                closest_indices = np.argsort(distances)[:self.num_winners]
                
                # Update winners: primary winner gets full lr, secondary winners get penalized lr
                for rank, c_idx in enumerate(closest_indices):
                    effective_lr = lr / (rank + 1)
                    self.centers[c_idx] += effective_lr * (x_k - self.centers[c_idx])
                    
        self._compute_widths()

    def _compute_widths(self):
        """
        Sets the width of each RBF unit to the distance to its nearest neighboring center.
        """
        self.widths = np.zeros(self.num_centers)
        for i in range(self.num_centers):
            distances = np.linalg.norm(self.centers - self.centers[i], axis=1)
            nearest_dist = np.sort(distances)[1]
            self.widths[i] = max(nearest_dist, 1e-3)

    def _rbf_activation(self, x, c, w):
        # N-dimensional Gaussian activation
        return np.exp(- (np.linalg.norm(x - c)**2) / (2 * w**2))

    def forward(self, X):
        X = np.atleast_2d(X)
        activations = np.array([[self._rbf_activation(x, c, w) 
                                 for c, w in zip(self.centers, self.widths)] for x in X])
        return np.dot(activations, self.weights)

    def fit_weights_delta(self, X, Y, epochs=50, lr=0.1):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y) 
        
        # Initialize weights randomly (Shape accommodates multi-dimensional outputs like ballistics)
        self.weights = np.random.randn(self.num_centers, Y.shape[1])
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            for idx in indices:
                x_k, y_k = X[idx], Y[idx]
                
                # Calculate phi (activations for single point)
                phi = np.array([self._rbf_activation(x_k, c, w) for c, w in zip(self.centers, self.widths)])
                
                # Predict and calculate instantaneous error
                pred = np.dot(phi, self.weights)
                error = y_k - pred
                
                # Delta rule weight update: reshape for correct matrix broadcasting
                self.weights += lr * np.outer(phi, error)