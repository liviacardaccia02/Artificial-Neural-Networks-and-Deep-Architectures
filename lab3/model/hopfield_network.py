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
    
class SequentialHopfieldNetwork(HopfieldNetwork):

    def update(self, state, neuron_index):
        activation = np.dot(self.W[neuron_index], state)

        if activation >= 0:
            state[neuron_index] = 1
        else:
            state[neuron_index] = -1

        return state

    def recall(self, state, iterations, record_every=100):
        state = state.copy()
        history = [state.copy()]

        n_neurons = len(state)

        for iteration in range(1, iterations + 1):

            # Randomly select one neuron
            neuron = np.random.randint(n_neurons)

            # Update only that neuron
            state = self.update(state, neuron)

            # Save state every N iterations
            if iteration % record_every == 0:
                print(f"Iteration {iteration}: Energy = {self.energy(state)}")
                history.append(state.copy())

        return state, history
    
    def is_stable(self, pattern):
        """
        Check whether every neuron remains unchanged
        when updated individually.
        """
        for neuron_index in range(self.n_neurons):

            state = pattern.copy()
            old_value = state[neuron_index]

            state = self.update(state, neuron_index)

            if state[neuron_index] != old_value:
                return False

        return True
    
    def energy(self, state):
        """
        Calculate the energy of the Hopfield system for a given state.
        """
        state = np.asarray(state)
        energy = 0
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                energy -= self.W[i, j] * state[i] * state[j]
        return energy
    
class SequentialHopfieldNetwork2:
    def __init__(self, patterns):
        self.patterns = np.asarray(patterns)
        self.n_neurons = patterns.shape[1]
        self.W = np.zeros((self.n_neurons, self.n_neurons))
        for p in patterns:
            self.W += np.outer(p, p)
            
        # Nota: per la capacità ottimale nel modello sequenziale è consigliato 
        # rimuovere le auto-connessioni impostando la diagonale a zero.
        # np.fill_diagonal(self.W, 0)

    def update(self, state):
        """
        Nel modello Hopfield originale gli stati vengono aggiornati uno alla volta, 
        permettendo a ciascuno di essere influenzato dagli altri stati che potrebbero 
        aver cambiato segno nei passi di aggiornamento precedenti.
        """
        new_state = np.asarray(state).copy()
        
        # Le unità vengono selezionate casualmente[cite: 1]
        indices = np.random.permutation(self.n_neurons)
        
        for i in indices:
            # Calcolo l'attivazione solo per il neurone i-esimo basandomi sullo stato corrente
            activation = np.dot(self.W[i], new_state)
            new_state[i] = 1 if activation >= 0 else -1
            
        return new_state

    def recall(self, state, iterations=100, record_every=10):
        state = np.asarray(state).copy()
        history = [state.copy()]

        for _ in range(iterations):
            new_state = self.update(state)
            if _ % record_every == 0:
                history.append(new_state.copy())
                print(f"Energy at iteration {_}: {self.energy(new_state)}")

            # Controllo di convergenza
            '''if np.array_equal(new_state, state):
                print(f"Converged after {_} iterations.")
                return new_state, history'''

            state = new_state

        return state, history

    def is_stable(self, pattern):
        """
        Un pattern è stabile se l'aggiornamento simultaneo teorico 
        non produrrebbe alcun cambiamento in nessuno dei neuroni.
        """
        activation = self.W @ pattern
        expected_state = np.where(activation >= 0, 1, -1)
        return np.array_equal(expected_state, pattern)
    
    def energy(self, state):
        """
        Calcola l'energia del sistema Hopfield per uno stato dato.
        
        """
        state = np.asarray(state)
        energy = 0
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                energy -= 0.5 * self.W[i, j] * state[i] * state[j]
        return energy