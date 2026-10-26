"""Vectorized multiclass neural network built with NumPy."""

import numpy as np


class WasteNeuralNetwork:
    """MLP with architecture input -> hidden1 -> hidden2 -> output.

    All rows in an input matrix are a mini-batch. This lets matrix multiplication
    calculate neuron activations for many images at once.
    """

    def __init__(self, input_size: int, hidden_sizes: tuple[int, int] = (64, 32), output_size: int = 3, seed: int = 42):
        generator = np.random.default_rng(seed)
        first_hidden, second_hidden = hidden_sizes
        self.parameters = {
            "W1": generator.normal(0, np.sqrt(2 / input_size), (input_size, first_hidden)),
            "b1": np.zeros((1, first_hidden)),
            "W2": generator.normal(0, np.sqrt(2 / first_hidden), (first_hidden, second_hidden)),
            "b2": np.zeros((1, second_hidden)),
            "W3": generator.normal(0, np.sqrt(2 / second_hidden), (second_hidden, output_size)),
            "b3": np.zeros((1, output_size)),
        }

    @staticmethod
    def relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0, values)

    @staticmethod
    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Vectorized feedforward pass, retaining values needed by backpropagation."""
        p = self.parameters
        z1 = features @ p["W1"] + p["b1"]
        a1 = self.relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = self.relu(z2)
        z3 = a2 @ p["W3"] + p["b3"]
        probabilities = self.softmax(z3)
        return probabilities, {"X": features, "Z1": z1, "A1": a1, "Z2": z2, "A2": a2}

    @staticmethod
    def compute_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
        return float(-np.mean(np.sum(targets * np.log(probabilities + 1e-12), axis=1)))

    def backward(self, probabilities: np.ndarray, targets: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the chain rule from Softmax loss back to every parameter."""
        p = self.parameters
        batch_size = len(targets)
        dz3 = (probabilities - targets) / batch_size
        gradients = {"W3": cache["A2"].T @ dz3, "b3": np.sum(dz3, axis=0, keepdims=True)}
        da2 = dz3 @ p["W3"].T
        dz2 = da2 * (cache["Z2"] > 0)
        gradients.update({"W2": cache["A1"].T @ dz2, "b2": np.sum(dz2, axis=0, keepdims=True)})
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (cache["Z1"] > 0)
        gradients.update({"W1": cache["X"].T @ dz1, "b1": np.sum(dz1, axis=0, keepdims=True)})
        return gradients

    def update_parameters(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        for name, gradient in gradients.items():
            self.parameters[name] -= learning_rate * gradient

    def fit(self, features: np.ndarray, targets: np.ndarray, epochs: int, batch_size: int, learning_rate: float, seed: int = 42) -> list[float]:
        generator = np.random.default_rng(seed)
        history: list[float] = []
        for epoch in range(epochs):
            order = generator.permutation(len(features))
            shuffled_x, shuffled_y = features[order], targets[order]
            batch_losses = []
            for start in range(0, len(features), batch_size):
                batch_x = shuffled_x[start : start + batch_size]
                batch_y = shuffled_y[start : start + batch_size]
                probabilities, cache = self.forward(batch_x)
                batch_losses.append(self.compute_loss(probabilities, batch_y))
                self.update_parameters(self.backward(probabilities, batch_y, cache), learning_rate)
            history.append(float(np.mean(batch_losses)))
            print(f"Epoch {epoch + 1:03d}/{epochs} | loss: {history[-1]:.4f} | train accuracy: {self.accuracy(features, targets):.3f}")
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities, _ = self.forward(features)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, features: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(self.predict(features) == np.argmax(targets, axis=1)))
