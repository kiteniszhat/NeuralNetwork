# NeuralNetwork: Deep Learning from Scratch

![Language](https://img.shields.io/badge/language-Python%203-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Implementation of a deep neural network library in pure Python (with NumPy). 

## Features

- **Custom Dense Layers**: Fully connected layers interacting via standard matrix operations.
- **Activation Functions**:
    - `ReLU`: Rectified Linear Unit for hidden layers ($f(x) = \max(0, x)$).
    - `Softmax`: For output probability distributions.
    - `Sigmoid`: For binary classification tasks.
- **Loss Functions**:
    - `CategoricalCrossEntropy`: Measures divergence between predicted probabilities and true labels.
    - `MeanSquaredError`: Standard regression loss.
- **Optimizers**:
    - **Stochastic Gradient Descent (SGD)**: Implements Momentum ($\mu$) and Learning Rate Decay for stable convergence.
- **Visualization**: Built-in tools to track Loss and Accuracy during training.

## Mathematical Core

This library implements backpropagation manually. For a standard Dense Layer with inputs $X$, weights $W$, and biases $B$:

1.  **Forward Pass**: 
    $$Z = X \cdot W + B$$
    
2.  **Backward Pass (Gradients)**:
    -   Error w.r.t Inputs: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot W^T$
    -   Error w.r.t Weights: $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Z}$
    -   Error w.r.t Biases: $\frac{\partial L}{\partial B} = \sum \frac{\partial L}{\partial Z}$

## Project Structure

```text
NeuralNetwork/
├── DenseLayer.py       # Core layer logic (Forward/Backward)
├── ActivationFunc.py   # ReLU, Softmax, Sigmoid
├── LossFunc.py         # CrossEntropy, MSE
├── Optimizers.py       # SGD with Momentum/Decay
├── Accuracy.py         # Metric calculation
├── mnist.ipynb         # (Demo) MNIST Digit Classification (~97% Acc)
├── example.ipynb       # (Demo) Spiral Data Classification
├── example2.ipynb      # (Demo) Advanced Training Loop
└── requirements.txt    # Dependencies (numpy, matplotlib, etc.)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/NeuralNetwork.git
   cd NeuralNetwork
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Train a model on the MNIST dataset to recognize handwritten digits:

1.  Open `mnist.ipynb` in Jupyter/VS Code.
2.  Run all cells.
3.  Observe the Training Loss/Accuracy plots and final Test evaluation.

**Code Snippet:**

```python
from DenseLayer import DenseLayer
from ActivationFunc import ReLU, SoftmaxCategoricalCrossEntropy
from Optimizers import StochasticGradientDescent

# 1. Architecture
dense1 = DenseLayer(784, 64)  # Input -> Hidden
activation1 = ReLU()
dense2 = DenseLayer(64, 10)   # Hidden -> Output
loss_activation = SoftmaxCategoricalCrossEntropy()

# 2. Optimizer
optimizer = StochasticGradientDescent(learning_rate=0.1, momentum=0.9)

# 3. Training Loop (Pseudo-code)
# ... inside loop ...
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

# Backward
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.derivative_inputs)
activation1.backward(dense2.derivative_inputs)
dense1.backward(activation1.derivative_inputs)

# Update
optimizer.update_params(dense1)
optimizer.update_params(dense2)
```

## Performance Results

### MNIST Classification
- **Architecture**: 784 -> 64 (ReLU) -> 10 (Softmax)
- **Epochs**: 10
- **Test Accuracy**: **~99%**
- **Test Loss**: ~0.04


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
