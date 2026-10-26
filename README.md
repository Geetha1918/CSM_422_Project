# Smart Waste Classification Using Deep Learning

A course-focused neural-network project that classifies waste images as **biodegradable**, **recyclable**, or **non-recyclable**. The model is implemented with NumPy so the mathematics behind vectorized feedforward propagation, cross-entropy loss, gradient descent, and backpropagation stays visible.

## Real-life problem

Incorrect waste segregation contaminates recyclable material and increases landfill waste. This project demonstrates how a camera-based classifier could suggest the correct disposal category before a person places an item in a bin. It is an educational prototype, not an automated waste-sorting system.

## Course-outcome coverage

| Course outcome / unit topic | Where it appears |
| --- | --- |
| Binary perceptron | `src/perceptron.py` trains recyclable vs. non-recyclable baseline |
| Multiclass classification | `src/neural_network.py` predicts 3 waste categories using Softmax |
| Image dimensions and information flow | Images are resized to 32 x 32, flattened to 1024 values, then flow through `1024 -> 64 -> 32 -> 3` |
| Parameters and hyperparameters | Weights/biases are parameters; learning rate, epochs, batch size and hidden sizes are CLI hyperparameters |
| Activation functions | ReLU in hidden layers, Softmax at output, Sigmoid in binary perceptron |
| Feedforward / vectorization | Matrix operations in `forward()` process an entire batch together |
| Loss minimization | Categorical cross-entropy is calculated in `compute_loss()` |
| Gradient descent and backpropagation | `backward()` calculates gradients; `update_parameters()` applies gradient descent |
| Training in batches | `fit()` shuffles data and trains in mini-batches |

## Network flow

```text
Waste image (32 x 32 grayscale)
           |
           v
Flatten to 1024 pixel values
           |
           v
Dense layer: 1024 -> 64, ReLU
           |
           v
Dense layer: 64 -> 32, ReLU
           |
           v
Dense layer: 32 -> 3, Softmax
           |
           v
Biodegradable / Recyclable / Non-recyclable
```

For a batch of `m` images, input `X` has dimension `(m, 1024)`. The first layer uses `Z1 = XW1 + b1`, so its output is `(m, 64)`. This is the vectorized feedforward implementation.

## Download and prepare the dataset

The repository includes a script that downloads the public **TrashNet** dataset (about 41 MB) and makes a reproducible 80/20 train/test split. It maps its six original categories into the three categories used by this project:

| TrashNet source class | Project class |
| --- | --- |
| cardboard, paper | biodegradable |
| glass, metal, plastic | recyclable |
| trash | non-recyclable |

Run this once, after installing the requirements:

```bash
python download_dataset.py --data-dir data
```

The script downloads images locally but does not commit them to GitHub, keeping the repository small and respecting the source dataset's distribution. TrashNet was created by Gary Thung and Mindy Yang; cite its [original repository](https://github.com/garythung/trashnet) in your report.

## Dataset layout

The download script creates this structure automatically. You may instead place your own photographed waste images (`.jpg`, `.jpeg`, `.png`) into it.

```text
data/
  train/
    biodegradable/
    recyclable/
    non_recyclable/
  test/
    biodegradable/
    recyclable/
    non_recyclable/
```

Suggested examples: biodegradable = food scraps, leaves, paper; recyclable = plastic bottles, cans, clean glass; non-recyclable = wrappers, contaminated plastic, mixed waste. Aim for at least 100 images per class for a demonstration; keep classes approximately balanced.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_dataset.py --data-dir data
```

## Run training

```bash
python train.py --data-dir data --epochs 50 --batch-size 16 --learning-rate 0.05
```

Training creates `outputs/training_history.png`, `outputs/confusion_matrix.png`, and `outputs/model_weights.npz`.

## Viva-ready explanation

1. A **perceptron** combines pixel inputs with weights and a bias. Its sigmoid output represents the probability of recyclable waste in the binary baseline.
2. The multiclass network uses **three output neurons**. Softmax converts their scores into probabilities that add to one.
3. In **forward propagation**, images move from input pixels through weighted hidden layers to output probabilities.
4. **Categorical cross-entropy** measures the difference between predicted probabilities and true labels.
5. In **backpropagation**, the loss gradient flows backward to calculate how every weight and bias contributed to the error.
6. **Gradient descent** subtracts a small multiple of each gradient from its parameter, reducing the loss over time.
7. The project uses **mini-batches**: several images are processed together, which is faster and gives more stable updates than one image at a time.

## Limitations and responsible use

Waste acceptance rules differ across locations. A real deployment would need many more diverse images, local recycling rules, human review for uncertain predictions, and safety procedures for hazardous waste.
