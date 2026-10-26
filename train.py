"""Train and evaluate the Smart Waste Classification neural-network project."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import CLASS_NAMES, describe_split, load_split, one_hot
from src.neural_network import WasteNeuralNetwork
from src.perceptron import BinaryPerceptron


def save_training_plot(losses: list[float], output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, color="#2a9d8f")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical cross-entropy loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()


def save_confusion_matrix(actual: np.ndarray, predicted: np.ndarray, output_dir: Path) -> None:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for truth, guess in zip(actual, predicted):
        matrix[truth, guess] += 1
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Predicted label", ylabel="True label", title="Test Confusion Matrix")
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def run_binary_baseline(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> None:
    """Show Unit-I binary perceptron with recyclable vs. non-recyclable images only."""
    keep_train = np.isin(train_y, [1, 2])
    keep_test = np.isin(test_y, [1, 2])
    binary_train_y = (train_y[keep_train] == 1).astype(int)
    binary_test_y = (test_y[keep_test] == 1).astype(int)
    model = BinaryPerceptron(train_x.shape[1])
    model.fit(train_x[keep_train], binary_train_y)
    accuracy = np.mean(model.predict(test_x[keep_test]) == binary_test_y)
    print(f"Binary perceptron baseline (recyclable vs non-recyclable) accuracy: {accuracy:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a vectorized waste-image neural network.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/ and test/ folders.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    arguments = parser.parse_args()

    data_dir = Path(arguments.data_dir)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    train_x, train_y = load_split(data_dir / "train")
    test_x, test_y = load_split(data_dir / "test")
    print("Training split:", describe_split(train_x, train_y))
    print("Test split:", describe_split(test_x, test_y))
    run_binary_baseline(train_x, train_y, test_x, test_y)

    network = WasteNeuralNetwork(input_size=train_x.shape[1])
    history = network.fit(train_x, one_hot(train_y, 3), arguments.epochs, arguments.batch_size, arguments.learning_rate)
    predictions = network.predict(test_x)
    test_accuracy = np.mean(predictions == test_y)
    print(f"Multiclass test accuracy: {test_accuracy:.3f}")
    save_training_plot(history, output_dir)
    save_confusion_matrix(test_y, predictions, output_dir)
    np.savez(output_dir / "model_weights.npz", **network.parameters)
    print(f"Saved model and graphs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
