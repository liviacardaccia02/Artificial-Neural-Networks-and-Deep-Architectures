import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        self.patterns = np.asarray(patterns)
        self.n_neurons = patterns.shape[1]
        self.W = np.zeros((self.n_neurons, self.n_neurons))
        for p in patterns:
            self.W += np.outer(p, p)

    def update(self, state):
        state = np.asarray(state)

        activation = self.W @ state
        new_state = np.where(activation >= 0, 1, -1)

        return new_state

    def recall(self, state, max_iterations=100):
        state = np.asarray(state).copy()

        history = [state.copy()]

        for _ in range(max_iterations):
            new_state = self.update(state)
            history.append(new_state.copy())

            # Stable fixed point
            if np.array_equal(new_state, state):
                return new_state, history, True

            state = new_state

        return state, history, False

    def is_stable(self, pattern):
        return np.array_equal(self.update(pattern), pattern)