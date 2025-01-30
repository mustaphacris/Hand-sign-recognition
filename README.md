# Sign Language Recognition using Deep Learning

## Project Overview
This project focuses on recognizing sign language gestures using deep learning techniques. It employs **MobileNetV2** as a feature extractor, along with **MediaPipe Hands** for hand landmark detection. The model is trained to classify different sign language gestures and can be used for real-time sign language interpretation via webcam.

## Features
- **Data Augmentation**: Uses **ImageDataGenerator** to create variations of training images to improve model generalization.
- **Hand Landmark Detection**: Utilizes **MediaPipe Hands** to detect key points of the hand.
- **Deep Learning Model**: Implements **MobileNetV2** for feature extraction and classification.
- **Dataset Balancing**: Ensures that all classes have an equal number of samples.
- **Real-time Detection**: Detects and classifies sign language gestures in real time using a webcam.

## Technologies Used
- Python
- OpenCV
- TensorFlow/Keras
- MediaPipe
- Matplotlib
- Pandas
- NumPy
- SciKit-Learn

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/sign-language-recognition.git
   cd sign-language-recognition
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure that your dataset is stored in the correct directory as defined in the script.

## Training the Model
Run the following command to train the model:
```sh
python train.py
```

This script will:
- Load and preprocess images
- Apply data augmentation and landmark detection
- Train a **MobileNetV2-based** model with early stopping and learning rate reduction
- Save the trained model

## Running Real-time Detection
Once the model is trained, you can use your webcam for real-time sign recognition:
```sh
python real_time_detection.py
```
Press 'q' to exit the webcam window.

## Results
The training process includes visualization of accuracy and loss curves, helping to analyze model performance. The model stops training when it reaches a target accuracy of **98%** or a loss below **0.2**.

## Future Improvements
- Expand dataset to include more sign language gestures.
- Optimize real-time processing for better performance.
- Implement gesture-to-text conversion for improved accessibility.

### Author:
 akhssaysaid023@gmail.com
 mustaphalahmer090@gmail.com

