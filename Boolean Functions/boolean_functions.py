import numpy as np
from itertools import product

class BooleanFunction:
    def __init__(self, n):
        self.n = n
        self.functions = self.generate_all_functions() if n <= 3 else self.generate_10000_functions()

    def generate_all_functions(self):
        inputs = list(product([-1, 1], repeat=self.n))
        functions = []
        for outputs in product([-1, 1], repeat=len(inputs)):
            functions.append((inputs, outputs))
        return functions
    
    def generate_10000_functions(self):
        inputs = list(product([-1, 1], repeat=self.n))  
        functions = set()  

        while len(functions) < 10000:  
            outputs = tuple(np.random.choice([-1, 1], len(inputs)))  
            functions.add((tuple(map(tuple, inputs)), outputs))  

        return list(functions)  

class Perceptron:
    def __init__(self, n, eta=0.05):
        self.n = n
        self.eta = eta
        self.weights = np.random.normal(0, 1/np.sqrt(n), n)
        self.threshold = 0

    def predict(self, x):
        b = np.dot(self.weights, x) - self.threshold
        return 1 if b >= 0 else -1

    def train(self, X, y, epochs=20):
        for _ in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                update = self.eta * (y[i] - prediction)
                self.weights += update * np.array(X[i])
                self.threshold -= update

    def is_linearly_separable(self, X, y):
        self.train(X, y, epochs=20)
        predictions = [self.predict(x) for x in X]
        return np.array_equal(predictions, y)

def evaluate_linear_separability(n):
    boolean_function = BooleanFunction(n)
    perceptron = Perceptron(n)
    linearly_separable_count = 0

    for inputs, outputs in boolean_function.functions:
        if perceptron.is_linearly_separable(inputs, outputs):
            linearly_separable_count += 1

    total_functions = len(boolean_function.functions)
    return linearly_separable_count, total_functions

if __name__ == "__main__":
    # For n = 2
    separable_count, total = evaluate_linear_separability(2)
    print(f"For n=2: {separable_count}/{total} functions are linearly separable")

    # For n = 3
    separable_count, total = evaluate_linear_separability(3)
    print(f"For n=3: {separable_count}/{total} functions are linearly separable")

    # For n = 4
    separable_count, total = evaluate_linear_separability(4)
    print(f"For n=4: {separable_count}/{total} functions are linearly separable")

    # For n = 5
    separable_count, total = evaluate_linear_separability(5)
    print(f"For n=5: {separable_count}/{total} functions are linearly separable")