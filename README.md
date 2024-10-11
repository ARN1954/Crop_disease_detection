
# Crop Disease Detection 🌾🤖

## Overview
This project focuses on detecting various crop diseases using Convolutional Neural Networks (CNNs). By leveraging deep learning techniques and image classification, the model can identify diseases from leaf images, helping farmers and agronomists monitor and manage crops more efficiently. The project can be used for early disease detection to mitigate potential agricultural losses and promote healthy crop yields.

### Key Features:
- **Deep Learning-Based Disease Detection**: Uses CNN architecture for classifying crop diseases from images.
- **Multi-Disease Support**: Trained to detect a variety of common crop diseases like bacterial spots, late blight, leaf mold, etc.
- **Image Preprocessing Pipeline**: Implements resizing, augmentation, normalization, and other preprocessing techniques to improve model accuracy.
- **Easy Deployment**: Can be deployed as a web or mobile app to assist farmers in real-time disease detection in the field.

## Dataset
The dataset used for this project consists of thousands of labeled images of diseased and healthy crops. Common datasets for plant disease classification include:
- [PlantVillage](https://www.kaggle.com/emmarex/plantdisease), which has 54,000+ images across 38 different classes.

The dataset contains labeled images for different crops, such as:
- Corn
- Tomato
- Potato

Each crop category has both **healthy** and **diseased** labels, making it a multi-class classification problem.

## Model Architecture
The project uses a **Convolutional Neural Network (CNN)** to classify the images. The CNN is built with the following components:
- **Convolutional Layers**: Extract features from the images.
- **MaxPooling Layers**: Downsample feature maps to reduce dimensionality.
- **Dropout**: Prevents overfitting by randomly setting activations to zero during training.
- **Fully Connected Layers**: Classify the image based on extracted features.

We have used pre-trained models like **VGG16** and **ResNet50** for transfer learning to improve accuracy and reduce training time.

## Workflow
1. **Data Preprocessing**: 
   - Image resizing and normalization.
   - Data augmentation (rotation, flipping, zooming, etc.) for generalization.

2. **Model Training**: 
   - The CNN is trained on labeled leaf images with various disease classes.
   - Loss function: `Categorical Cross-Entropy`.
   - Optimizer: `Adam` for faster convergence.

3. **Evaluation**:
   - The trained model is evaluated on the test set to calculate accuracy, precision, recall, and F1 score.
   - Confusion matrix and classification reports are generated for detailed analysis.

4. **Deployment**: 
   - The model can be exported and deployed using TensorFlow, PyTorch, or converted into a format suitable for mobile deployment using TensorFlow Lite.

## Installation 🛠️

### Requirements
Make sure to install the following dependencies before running the code:

- Python 3.x
- TensorFlow or PyTorch (depending on implementation)
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

You can install the necessary Python packages by running:


pip install -r requirements.txt


https://github.com/user-attachments/assets/79b4aea9-904b-4ca0-8b5f-4bf14b67f911


## Results 📊

After training, the model achieves an accuracy of over 90% on the validation set. The following evaluation matrics are used:

| Matric        | Value |
|---------------|-------|
| Accuracy      | 92%   |
| Precision     | 91%   |
| Recall        | 90%   |
| F1 Score      | 90.5% |

Sample predictions:

| Input Image | Prediction | Confidence |
|-------------|------------|------------|
| ![Leaf Image](https://github.com/user-attachments/assets/2b90d65f-6f9f-490b-b600-49350c45b462) | Tomato Yellow Leaf | 95.6% |
| ![Leaf Image](https://github.com/user-attachments/assets/5f359beb-9555-41f1-8590-7de147303f60) | Potato Early bright | 98.51% |
| ![Leaf Image](https://github.com/user-attachments/assets/707303e5-7f26-4521-a87f-1cbbc98268d9) | Corn Gray Spot | 100% |
| ![Leaf Image](https://github.com/user-attachments/assets/6cfbc8ad-1119-428c-93f4-1e952c3d03af) | Corn Early Bright | 98.51% |




