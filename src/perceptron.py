"""Binary sigmoid perceptron baseline: recyclable versus non-recyclable."""

import numpy as np


class BinaryPerceptron:
    """A single neuron trained by batch gradient descent.

    It is retained as an explicit Unit-I baseline before the multiclass network.
    """

    def __init__(self, input_size: int, learning_rate: float = 0.1, seed: int = 42):
        generator = np.random.default_rng(seed)
        self.weights = generator.normal(0, 0.01, input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.sigmoid(features @ self.weights + self.bias)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= 0.5).astype(int)

    def fit(self, features: np.ndarray, labels: np.ndarray, epochs: int = 30) -> list[float]:
        """Train using sigmoid derivative and gradient descent."""
        history = []
        for _ in range(epochs):
            probabilities = self.predict_proba(features)
            error = probabilities - labels
            gradient_w = features.T @ error / len(features)
            gradient_b = float(np.mean(error))
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b
            loss = -np.mean(labels * np.log(probabilities + 1e-12) + (1 - labels) * np.log(1 - probabilities + 1e-12))
            history.append(float(loss))
        return history
