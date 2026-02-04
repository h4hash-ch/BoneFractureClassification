# Bone Fracture Classification

Deep learning project for automated bone fracture detection from X-ray images using transfer learning and custom CNN architectures. Compares three models to identify the most effective approach for medical image classification.

## Overview

This notebook implements and evaluates three convolutional neural network architectures for binary classification of bone fractures from X-ray images. The project uses transfer learning with ResNet50 and VGG16, alongside a custom CNN built from scratch, to determine which approach yields the best diagnostic accuracy.

## Results

Three models evaluated on the Bone Fracture Dataset:

| Model | Architecture | Training Strategy |
|-------|--------------|-------------------|
| ResNet50 | Transfer Learning | Pre-trained on ImageNet, fine-tuned last 10 layers |
| VGG16 | Transfer Learning | Pre-trained on ImageNet, last 10 layers trainable |
| Custom CNN | Built from Scratch | 3 convolutional blocks with batch normalization |

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Run the notebook to see detailed performance comparison and training curves for all three models.

## Prerequisites

Required Libraries:
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn
- Pillow
- tqdm

## Dataset

**Source:** [Bone Fracture Dataset on Kaggle](https://www.kaggle.com/)  
**Classes:** Binary classification (Fractured / Not Fractured)  
**Format:** X-ray images

Expected directory structure:
```
data/
  train/
    fractured/
    not_fractured/
  validation/
  test/
```

## How to Run

1. Download `BoneFractureClassification.ipynb` from this repository

2. Get the Dataset:
   - Download from [Kaggle Bone Fracture Dataset](https://www.kaggle.com/)
   - Organize images in the directory structure shown above

3. Open in Google Colab or Jupyter:
   - For Colab: Go to [Google Colab](https://colab.research.google.com/)
   - Upload the notebook (File → Upload notebook)
   - Mount Google Drive if dataset is stored there
   - Update dataset paths as needed
   - Run all cells (Runtime → Run all)

## What's Inside

### 1. Data Preparation
Loads and organizes the bone fracture dataset, performs data cleaning, and sets up directory structure for training, validation, and test sets.

### 2. Exploratory Data Analysis
Visualizes sample X-ray images from both classes (fractured and not fractured), analyzes class distribution, and examines dataset characteristics.

### 3. Data Augmentation
Configures ImageDataGenerator with augmentation techniques to improve model robustness and prevent overfitting on limited medical imaging data.

### 4. ResNet50 Model (Transfer Learning)
Implements ResNet50 with pre-trained ImageNet weights. Custom top layers added for binary classification. Two-phase training: initial training with frozen base, followed by fine-tuning of the last 10 layers.

### 5. VGG16 Model (Transfer Learning)
Implements VGG16 with pre-trained ImageNet weights. Last 10 layers unfrozen from the start for trainable parameters. Custom fully connected layers for fracture classification.

### 6. Custom CNN Model
Built-from-scratch architecture with three convolutional blocks. Each block includes convolution, batch normalization, max pooling, and dropout for regularization. Fully connected layers with additional dropout for final classification.

### 7. Model Evaluation
Comprehensive evaluation of all three models using confusion matrices, classification reports, and performance visualizations. Training and validation loss curves plotted to assess learning progression.

### 8. Performance Comparison
Side-by-side comparison of final training accuracy, validation accuracy, and loss metrics. Bar charts visualize relative performance of ResNet50, VGG16, and Custom CNN.

## Model Architectures

### ResNet50 (Transfer Learning)
- Base: ResNet50 pre-trained on ImageNet (frozen initially)
- Custom Layers: GlobalAveragePooling2D → Dense(256, ReLU) → Dense(1, Sigmoid)
- Training: 10 epochs initial + 10 epochs fine-tuning
- Optimizer: Adam (lr=0.0001, fine-tune lr=0.00001)

### VGG16 (Transfer Learning)
- Base: VGG16 pre-trained on ImageNet (last 10 layers trainable)
- Custom Layers: Flatten → Dense(256, ReLU) → Dense(1, Sigmoid)
- Training: 10 epochs
- Optimizer: Adam (lr=0.0001)

### Custom CNN
- Architecture: 3 Conv blocks (32→64→128 filters)
- Regularization: Batch Normalization + Dropout (0.25 per block, 0.5 final)
- Fully Connected: Dense(512, ReLU) → Dense(1, Sigmoid)
- Training: 10 epochs with EarlyStopping
- Optimizer: Adam (lr=0.0001)

## Data Processing Pipeline

**Image Preprocessing:**
- Resize: All images scaled to 224×224 pixels
- Normalization: Pixel values rescaled to [0, 1] range
- Batch Size: 32 images per batch
- Augmentation: Applied via ImageDataGenerator for training set

**Data Generators:**
- Training: Shuffled batches with augmentation
- Validation: Shuffled batches without augmentation
- Test: Sequential batches for consistent evaluation

## Key Insights

- Transfer learning leverages pre-trained weights from ImageNet, potentially improving performance on limited medical data
- Fine-tuning strategy (ResNet50) allows adaptation of learned features to fracture detection
- Custom CNN provides baseline performance without pre-trained knowledge
- Batch normalization and dropout are critical for preventing overfitting in medical imaging
- Data augmentation helps models generalize better on X-ray variations
- Model comparison reveals trade-offs between complexity, training time, and accuracy

## Technical Details

**Image Specifications:**
- Input Shape: (224, 224, 3)
- Color Mode: RGB
- Rescaling: 1./255 normalization

**Training Configuration:**
- Loss Function: Binary Cross-Entropy
- Metrics: Accuracy
- Epochs: 10 (ResNet50: 20 with fine-tuning)
- Callbacks: EarlyStopping (Custom CNN only)

**Performance Evaluation:**
- Confusion Matrix: True/False Positives and Negatives
- Classification Report: Precision, Recall, F1-Score per class
- Accuracy: Overall correct predictions
- Loss Curves: Training and validation progression

## Use Cases

This model can assist in:
- Preliminary fracture screening in emergency departments
- Educational tools for medical students
- Research in automated medical image analysis
- Benchmark for comparing CNN architectures on medical data

## Future Improvements

- Expand dataset with more diverse X-ray images
- Implement additional architectures (EfficientNet, DenseNet)
- Apply advanced augmentation techniques (Mixup, CutMix)
- Add explainability visualizations (Grad-CAM)
- Perform extensive hyperparameter tuning
- Cross-validation for robust performance estimates
- Multi-class classification (fracture types)

## Author Contact

**Hashim Choudhry**
- GitHub: [@h4hash-ch](https://github.com/h4hash-ch)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/hashim-choudhry/)

## Disclaimer

This project is for educational and research purposes only. Not intended for clinical diagnosis or medical decision-making. Always consult qualified medical professionals for fracture diagnosis and treatment.
