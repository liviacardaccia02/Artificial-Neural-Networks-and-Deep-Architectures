import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Perceptron(ABC):
    def __init__(self,
                 input_size, 
                 output_size, 
                 learning_rate=0.1, 
                 activation_function=None
                ):
        
        self.weights = np.random.randn(output_size, input_size+1)
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.mean_squared_error = []
        self.misclassified_samples = []
        self.misclassified_samples_rate = []
    

    def train(self, X, T, epochs):
        for epoch in range(epochs):
            self.epoch_train(X, T)
        self.plot_space(X, T)
            
    @abstractmethod
    def epoch_train(self, X, T):
        pass
         
    def predict(self, x):
        return np.dot(self.weights, x)
    
    def plot_space(self, X, T):
        x = np.linspace(X[0].min(), X[0].max(), 20)
        y = -(self.weights[0, 0] * x + self.weights[0, 2]) / self.weights[0, 1]
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