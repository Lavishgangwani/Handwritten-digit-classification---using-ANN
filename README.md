---

# Handwritten Digit Classification using Deep Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview
This project demonstrates how to classify handwritten digits (0-9) using a deep learning model built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/). The model is trained on the **MNIST** dataset, which contains 70,000 grayscale images of handwritten digits, each 28x28 pixels in size. The goal is to accurately classify these images into their respective digit classes.

## Dataset
The dataset used is the popular **MNIST** dataset, which consists of:
- **Training data**: 60,000 images with corresponding labels.
- **Test data**: 10,000 images with corresponding labels.

Each image is a 28x28 pixel grayscale image, and the label represents the digit (0-9).

**Dataset Source:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## Model Architecture
The deep learning model is built using a Artifical Neural Network (aNN), a architecture for image classification tasks. The architecture includes:
1. **Input Layer:** 28x28 grayscale images.
2. **Hidden Layers:** 2 layers of (128,32 Neurons) to extract features.
3. **Fully Connected Layers (Dense):** Perform classification based on the extracted features.
4. **Output Layer:** Softmax activation to predict probabilities for each of the 10 digit classes.

The detailed architecture:
- **Dense layer 1**: 128 neurons, ReLU activation.
- **Dense layer 2**: 32 neurons, ReLU activation
- **Output layer**: 10 units, Softmax activation for multiclass classification.

## Installation
To run this project, you will need to install the following dependencies:

1. Clone this repository:
    ```bash
    git clone https://github.com/lavishgangwani/handwritten-digit-classification---using-ANN.git
    ```
   
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

   **Requirements:**
   - TensorFlow
   - Keras
   - NumPy
   - Matplotlib (for visualizing results)
   - Jupyter (if you want to run the notebook)

## Usage
You can use the provided Jupyter notebook to train and test the model.

**Jupyter Notebook**: open the `handwritten_digit_classification.ipynb` to run and modify the code interactively.


## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
