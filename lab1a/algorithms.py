import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Perceptron(ABC):
    def __init__(self,
                 input_size, 
                 output_size, 
                 learning_rate=0.1, 
                 activation_function=None,
                 has_bias=True
                ):
        
        self.weights = np.random.randn(output_size, input_size + (1 if has_bias else 0))
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.has_bias = has_bias
        self.mean_squared_error = []
        self.misclassified_samples = []
        self.misclassified_samples_rate = []
        self.misclassified_samples_A = []
        self.misclassified_samples_B = []
        self.misclassified_samples_A_rate = []
        self.misclassified_samples_B_rate = []
    

    def train(self, X, T, epochs):
        self.reset_metrics()
        self.weights = np.random.randn(1, 2 + (1 if self.has_bias else 0))
        for epoch in range(epochs):
            self.epoch_train(X, T)
        self.plot_space(X, T)
            
    @abstractmethod
    def epoch_train(self, X, T):
        pass
         
    def predict(self, x):
        return np.dot(self.weights, x)
    
    def get_first_zero_misclassification_epoch(self):
        for epoch, misclassified in enumerate(self.misclassified_samples):
            if misclassified == 0:
                return epoch
        return None
    
    def reset_metrics(self):
        self.mean_squared_error = []
        self.misclassified_samples = []
        self.misclassified_samples_rate = []
        self.misclassified_samples_A = []
        self.misclassified_samples_B = []
        self.misclassified_samples_A_rate = []
        self.misclassified_samples_B_rate = []
    
    def plot_space(self, X, T):
        x = np.linspace(X[0].min(), X[0].max(), 20)
        y = -(self.weights[0, 0] * x + (self.weights[0, 2] if self.has_bias else 0)) / self.weights[0, 1]
        plt.scatter(X[0, T != 1], X[1, T != 1], label="Class B")
        plt.scatter(X[0, T == 1], X[1, T == 1], label="Class A")

        plt.plot(x, y, label="Decision boundary")

        padding = 0.1
        plt.xlim(X[0].min() - padding, X[0].max() + padding)
        plt.ylim(X[1].min() - padding, X[1].max() + padding)

        plt.legend()
        plt.show()
        
    def plot_metrics(self):
        epochs = range(len(self.mean_squared_error))
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.mean_squared_error, label='Mean Squared Error')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Mean Squared Error over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.misclassified_samples_rate, label='Misclassified Samples Rate', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassified Samples Rate')
        plt.title('Misclassified Samples Rate over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
class PerceptronRulePerceptron(Perceptron):
    def epoch_train(self, X, T):
        misclassified_counter = 0
        mean_squared_error = 0
        for x, t in zip(X.T, T):
            y = self.predict(x)
            y = self.activation_function(y)
            if y != t:
                misclassified_counter += 1
            error = t - y
            self.weights += self.learning_rate * error * x
            mean_squared_error += error ** 2
        self.misclassified_samples.append(misclassified_counter)
        self.misclassified_samples_rate.append(misclassified_counter / len(X.T))
        self.mean_squared_error.append(mean_squared_error / len(X.T))
        
class DeltaRulePerceptron(Perceptron):
    def epoch_train(self, X, T):
        misclassified_counter = 0
        mean_squared_error = 0
        for x, t in zip(X.T, T):
            y = self.predict(x)
            error = t - y
            y = self.activation_function(y)
            if y != t:
                misclassified_counter += 1
            self.weights += self.learning_rate * error * x
            
            mean_squared_error += error ** 2
        self.misclassified_samples.append(misclassified_counter)
        self.misclassified_samples_rate.append(misclassified_counter / len(X.T))
        self.mean_squared_error.append(mean_squared_error / len(X.T))


class DeltaRuleBatchPerceptron(Perceptron):
    def epoch_train(self, X, T):
        weight_update = np.zeros_like(self.weights)

        Y = self.predict(X)
        error = T - Y
        
        weight_update = self.learning_rate * np.dot(error, X.T) / X.shape[1]

        Y = self.activation_function(Y)
        misclassified_counter = np.sum(Y != T)
        
        self.weights += weight_update
        self.misclassified_samples.append(misclassified_counter)
        self.misclassified_samples_rate.append(misclassified_counter / X.shape[1])
        self.mean_squared_error.append(np.mean(error ** 2))
        
        self.misclassified_samples_A.append(np.sum((Y != T) & (T == 1)))
        self.misclassified_samples_B.append(np.sum((Y != T) & (T == -1)))
        self.misclassified_samples_A_rate.append(np.sum((Y != T) & (T == 1)) / np.sum(T == 1))
        self.misclassified_samples_B_rate.append(np.sum((Y != T) & (T == -1)) / np.sum(T == -1))
