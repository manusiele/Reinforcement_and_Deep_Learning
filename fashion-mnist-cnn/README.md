# Fashion-MNIST CNN Classification

A Convolutional Neural Network (CNN) implementation for classifying Fashion-MNIST dataset images using TensorFlow/Keras.

##  Project Overview

This project implements a deep learning solution for classifying fashion items from the Fashion-MNIST dataset into 10 categories. The model achieves high accuracy using a custom CNN architecture with regularization techniques.

##  Objective

Build and train a CNN to classify grayscale images (28x28 pixels) of fashion items into one of 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Model Architecture

### CNN Design:
- **Conv Layer 1:** 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling 1:** 2x2 pool size
- **Dropout 1:** 0.25 rate
- **Conv Layer 2:** 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling 2:** 2x2 pool size
- **Dropout 2:** 0.25 rate
- **Flatten Layer**
- **Dense Layer:** 128 units, ReLU activation
- **Dropout 3:** 0.5 rate
- **Output Layer:** 10 units, Softmax activation

### Architecture Rationale:
- **Two convolutional layers** capture hierarchical features (edges → patterns → objects)
- **MaxPooling** reduces spatial dimensions and provides translation invariance
- **Dropout** prevents overfitting at multiple stages
- **3x3 kernels** balance receptive field size and computational efficiency

## Dataset

- **Source:** Fashion-MNIST (built into TensorFlow/Keras)
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Image size:** 28x28 grayscale
- **Classes:** 10

##  Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Running on Google Colab

1. Upload `fashion_mnist_cnn.ipynb` to Google Colab
2. Run all cells sequentially
3. No additional setup required - all dependencies are pre-installed


## Training Details

- **Epochs:** 15
- **Batch size:** 128
- **Optimizer:** Adam
- **Loss function:** Categorical Cross-Entropy
- **Validation split:** 20%
- **Training time:** ~5-10 minutes on GPU

##  Results

### Expected Performance:
- **Test Accuracy:** ~90-92%
- **Training Accuracy:** ~93-95%
- **Test Loss:** ~0.25-0.30

### Evaluation Metrics:
- Accuracy and loss curves (training vs validation)
- Confusion matrix
- Classification report (precision, recall, F1-score per class)
- Sample predictions visualization

## Key Findings

### Overfitting Analysis:
- Monitor the gap between training and validation accuracy
- Dropout layers help prevent overfitting
- Early stopping can be implemented if validation loss plateaus

### Performance Insights:
- Some classes (e.g., Shirt vs T-shirt) are harder to distinguish
- Confusion matrix reveals common misclassifications
- Model performs well on distinct categories (Trouser, Bag, Sneaker)

##  Potential Improvements

1. **Data Augmentation:** Random rotations, shifts, and zooms
2. **Batch Normalization:** Faster convergence and better generalization
3. **Learning Rate Scheduling:** Adaptive learning rate reduction
4. **Deeper Architecture:** Add third convolutional block
5. **Early Stopping:** Prevent overfitting with patience-based stopping
6. **Ensemble Methods:** Combine multiple models for better predictions

## Project Structure

```
fashion-mnist-cnn/
│
├── fashion_mnist_cnn.ipynb    # Main notebook with complete implementation
├── README.md                   # Project documentation
└── fashion_mnist_cnn_model.h5 # Saved trained model (generated after training)
```

## Academic Context

**Course:** Deep Learning / Neural Networks  
**Task:** Practical Task 1 - CNN for Fashion-MNIST Classification  
**Marks:** 10  

##  References

- Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras API: https://keras.io/

##  License

This project is created for educational purposes.

##  Author

Created as part of a deep learning practical assignment.

---

**Note:** This implementation is designed for Google Colab but can be adapted for local execution with appropriate environment setup.
