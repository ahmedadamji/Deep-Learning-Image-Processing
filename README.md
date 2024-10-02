# Machine Learning and Deep Learning Algorithms for Image Classification and Processing

This project demonstrates the implementation of advanced AI/ML algorithms, including machine learning techniques like linear regression, clustering, and classification from scratch using numpy, alongside deep learning models such as auto-encoders and variational auto-encoders (VAE) built with PyTorch. The focus spans image processing, dimensionality reduction, and classification tasks on well-known datasets such as MNIST and Iris.

## Notebooks Included

1. **Machine Learning from Scratch**:
   - Implemented machine learning algorithms from scratch using numpy and matplotlib for tasks such as linear fitting, clustering (PCA, K-means), and classification (logistic regression, SVM).
   - Constructed neural networks from scratch for binary classification tasks using gradient descent and backpropagation.
   - Applied PCA for dimensionality reduction and K-means for clustering on Iris and MNIST datasets.

2. **Deep Learning for Image Processing**:
   - Developed convolutional auto-encoders (AE) and variational auto-encoders (VAE) for image denoising, generation, and reconstruction on datasets like MNIST and CelebA.
   - Implemented deep learning techniques for noise removal and image reconstruction using encoder-decoder architectures in PyTorch.
   - Integrated image processing techniques such as Fourier-based high-pass filtering, and Sobel edge detection.

## Key Features

- **Machine Learning Algorithms from Scratch**: Built linear classifiers, clustering models (K-means, PCA), and neural networks using numpy for direct control over every computation.
- **Deep Learning with PyTorch**: Implemented convolutional auto-encoders and variational auto-encoders to denoise images, generate new samples, and reconstruct input data.
- **Dimensionality Reduction and Feature Engineering**: Applied PCA and Fourier-based methods to extract and visualize critical features.
- **Neural Network Implementation**: Trained fully connected neural networks using backpropagation, with the models evaluated on classification tasks like MNIST.

## Datasets

- **MNIST**: A dataset of handwritten digits used to demonstrate image reconstruction, denoising, and classification tasks.
- **Iris**: A widely-used dataset for clustering, linear fitting, and classification tasks, including visualizing decision boundaries and PCA components.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ahmedadamji/Deep-Learning-Image-Processing.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebooks:
   ```
   jupyter notebook
   ```

## License
This project is licensed under the MIT License.
