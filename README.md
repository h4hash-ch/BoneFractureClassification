# Bone Fracture Classification

A deep learning project for automated bone fracture detection from X-ray images using transfer learning and custom CNN architectures.

## Overview
This project implements and compares three convolutional neural network architectures (ResNet50, VGG16, and a custom CNN) for binary classification of bone fractures. The notebook includes complete data preprocessing, exploratory data analysis, model training, evaluation, and performance comparison.

**Repository snapshot:**
- Notebook: [BoneFractureClassification.ipynb](https://github.com/h4hash-ch/BoneFractureClassification/blob/main/BoneFractureClassification.ipynb)

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Notebook Structure](#notebook-structure)
- [Models & Parameters](#models--parameters)
- [Data Processing](#data-processing)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Acknowledgements](#Acknowledgements)

## Features
- **Three model architectures**: ResNet50, VGG16 (transfer learning), and custom CNN
- **Complete pipeline**: Data cleaning, EDA, preprocessing, training, and evaluation
- **Data augmentation**: ImageDataGenerator for robust training
- **Comprehensive metrics**: Confusion matrices, classification reports, and performance visualizations
- **Model comparison**: Side-by-side evaluation of all three architectures

## Dataset
- **Source**: [Bone Fracture Dataset on Kaggle](https://www.kaggle.com/datasets/ahmedashrafahmed/bone-fracture)
- **Classes**: Binary classification (Fractured / Not Fractured)
- **Format**: X-ray images

The notebook expects images organized in the following structure:
```
data/
  train/
    fractured/
    not_fractured/
  validation/
  test/
```

## Notebook Structure

The notebook contains the following sections:

1. **Library Imports**: All necessary dependencies
2. **Data Cleaning**: Preprocessing and organizing the dataset
3. **EDA**: Exploratory analysis and visualization of the dataset
4. **Image Display**: Sample visualization of fracture and non-fracture images
5. **Data Generators**: ImageDataGenerator setup with augmentation
6. **ResNet50 Model**: Transfer learning implementation
7. **VGG16 Model**: Transfer learning implementation
8. **Custom CNN Model**: Custom architecture built from scratch
9. **Model Comparison**: Performance metrics and visualization

## Models & Parameters

### 1. ResNet50 (Transfer Learning)

**Architecture:**
```python
Base Model: ResNet50 (pre-trained on ImageNet)
- Weights: 'imagenet'
- Input Shape: (224, 224, 3)
- Include Top: False
- Base Layers: Frozen initially

Custom Top Layers:
- GlobalAveragePooling2D()
- Dense(256, activation='relu')
- Dense(1, activation='sigmoid')
```

**Training Parameters:**
- **Optimizer**: Adam
  - Initial Learning Rate: 0.0001
  - Fine-tuning Learning Rate: 0.00001 (last 10 layers unfrozen)
- **Loss Function**: binary_crossentropy
- **Metrics**: accuracy
- **Epochs**: 
  - Initial Training: 10 epochs
  - Fine-tuning: 10 epochs (20 total)
- **Fine-tuning Strategy**: Unfreeze last 10 layers after initial training

### 2. VGG16 (Transfer Learning)

**Architecture:**
```python
Base Model: VGG16 (pre-trained on ImageNet)
- Weights: 'imagenet'
- Input Shape: (224, 224, 3)
- Include Top: False
- Trainable Layers: Last 10 layers unfrozen

Custom Top Layers:
- Flatten()
- Dense(256, activation='relu')
- Dense(1, activation='sigmoid')
```

**Training Parameters:**
- **Optimizer**: Adam
  - Learning Rate: 0.0001
- **Loss Function**: binary_crossentropy
- **Metrics**: accuracy
- **Epochs**: 10
- **Fine-tuning Strategy**: Last 10 layers trainable from start

### 3. Custom CNN

**Architecture:**
```python
Input Shape: (224, 224, 3)

Convolutional Blocks:
Block 1:
  - Conv2D(32, (3,3), activation='relu')
  - BatchNormalization()
  - MaxPooling2D((2,2))
  - Dropout(0.25)

Block 2:
  - Conv2D(64, (3,3), activation='relu')
  - BatchNormalization()
  - MaxPooling2D((2,2))
  - Dropout(0.25)

Block 3:
  - Conv2D(128, (3,3), activation='relu')
  - BatchNormalization()
  - MaxPooling2D((2,2))
  - Dropout(0.25)

Fully Connected Layers:
  - Flatten()
  - Dense(512, activation='relu')
  - BatchNormalization()
  - Dropout(0.5)
  - Dense(1, activation='sigmoid')
```

**Training Parameters:**
- **Optimizer**: Adam
  - Learning Rate: 0.0001
- **Loss Function**: binary_crossentropy
- **Metrics**: accuracy
- **Epochs**: 10
- **Callbacks**: 
  - EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

## Data Processing

### Image Data Generator Configuration

```python
Image Size: (224, 224)
Batch Size: 32
Rescaling: 1./255 (pixel normalization to [0, 1])

Data Split:
- Training Set: flow_from_dataframe (shuffled)
- Validation Set: flow_from_dataframe (shuffled)
- Test Set: flow_from_dataframe (not shuffled)

Classification Mode: binary
```

### Preprocessing Pipeline
1. Image loading from dataframe
2. Resizing to 224×224 pixels
3. Pixel value normalization (0-1 range)
4. Batch generation for training

## Results

All three models are evaluated on test data with the following metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False positive and negative counts
- **Training/Validation Loss Curves**: Model learning progression

### Performance Comparison

The notebook generates a comparative analysis including:
- Final training and validation accuracy for each model
- Final training and validation loss for each model
- Side-by-side bar charts for visual comparison
- Detailed classification reports

Run the notebook to generate complete performance metrics and visualizations.

## Dependencies

Core libraries used:
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualization
- **OpenCV**: Image processing
- **Scikit-learn**: Machine learning utilities
- **Pillow**: Image handling
- **tqdm**: Progress tracking

## Usage

### Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook BoneFractureClassification.ipynb
```

2. Run cells sequentially from top to bottom

### Using Google Colab

1. Upload the notebook to Google Colab
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ahmedashrafahmed/bone-fracture) and organize it as shown above.
3. Mount Google Drive if storing the dataset there:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Update dataset paths as needed
4. Run all cells

### Model Training

Each model is trained independently in its respective section:

```python
# Train ResNet50
resnet_history = resnet_model.fit(
    train_generator, 
    epochs=10, 
    validation_data=val_generator
)

# Train VGG16
history = model.fit(
    train_generator, 
    epochs=10, 
    validation_data=val_generator
)

# Train Custom CNN
cnn_history = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping]
)
```

### Model Evaluation

```python
# Evaluate any model on test set
test_loss, test_accuracy = model.evaluate(test_generator)
predictions = model.predict(test_generator)
```

## Acknowledgements

- Dataset by [Ahmed Ashraf Ahmed](https://www.kaggle.com/datasets/ahmedashrafahmed/bone-fracture) on Kaggle
- Pre-trained models from TensorFlow/Keras Applications
- Medical imaging research community

---

**Disclaimer**: This project is for educational and research purposes only. Not intended for clinical diagnosis. Consult qualified medical professionals for medical decisions.
