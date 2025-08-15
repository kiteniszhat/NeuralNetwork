# NeuralNetwork

My implementation of a Neural Network from scratch in Python.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## About

This repository contains a basic, from-scratch implementation of a neural network in Python only with NumPy. The goal of this project is to predict MNIST dataset.

## Features

- **Dense Layers**: Fundamental building blocks for neural networks.
- **Activation Functions**: Includes common activation functions like ReLU, Sigmoid, and Softmax.
- **Loss Functions**: Supports loss calculations such as Categorical Cross-Entropy and Mean Squared Error.
- **Optimizers**: Implements optimizers like Stochastic Gradient Descent for model training (AdaGrad and Adam soon...).
- **Accuracy Calculation**: Tools to evaluate the performance of the neural network.
- **Dropout and other adjustments**: Work in progress.

## Project Structure

- `Accuracy.py`: Defines functions to calculate the accuracy of the model.
- `ActivationFunc.py`: Contains various activation functions used in neural network layers.
- `DenseLayer.py`: Implements the core dense (fully connected) layer of the neural network.
- `LossFunc.py`: Provides different loss functions to measure the error of the model's predictions.
- `Optimizers.py`: Includes optimization algorithms to update network weights and biases during training.
- `example.ipynb`: A Jupyter Notebook demonstrating a basic usage of the neural network components.
- `example2.ipynb`: Another Jupyter Notebook with backpropagation.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `LICENSE`: The license under which this project is distributed.
- `README.md`: This README file.

## Installation

To get started with this project, clone the repository and install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/NeuralNetwork.git
   cd NeuralNetwork
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   # source .venv/bin/activate    # On macOS/Linux
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can explore the neural network implementation and examples. The primary way to interact with the code is through the provided Jupyter notebooks.

To start Jupyter Notebook:

```bash
jupyter notebook
```

## Examples

Refer to the following Jupyter notebooks for practical examples:

- `example.ipynb`: [Link to example.ipynb](example.ipynb)
- `example2.ipynb`: [Link to example2.ipynb](example2.ipynb)

These notebooks demonstrate how to:
- Initialize dense layers.
- Apply activation functions.
- Define and use loss functions.
- Configure and use optimizers for training.
- Evaluate model accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
