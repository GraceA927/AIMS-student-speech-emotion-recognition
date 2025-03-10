# AIMS-student-speech-emotion-recognition using CNN


## Overview
This project focuses on Speech Emotion Recognition (SER) using a Convolutional Neural Network (CNN). The model is trained on a dataset collected at the African Institute for Mathematical Sciences (AIMS) during the 2024-2025 academic year. The dataset contains speech recordings labeled with three emotions: **happy, sad, and neutral**. The goal is to classify speech samples into these categories based on extracted audio features.

## Dataset
Source:Collected at AIMS Cameroon
Emotions: Happy, Sad, Neutral
Format: WAV audio files
Preprocessing: Audio feature extraction using Mel spectrograms, MFCCs, and data augmentation techniques

## Methodology
1. Data Preprocessing:
   Convert audio files into Mel spectrograms
   Normalize and augment data (pitch shifting, noise addition, time stretching)
2. Feature Extraction:
   - Extract Mel Frequency Cepstral Coefficients (MFCCs)
   - Use spectrogram representations as input for the CNN
3. Model Architecture:
   A CNN-based deep learning model
    Convolutional layers for feature extraction
   Fully connected layers for classification
   Dropout technique applied to reduce overfitting and improve generalization
5. Training:
   Optimized using categorical cross-entropy loss
   Adam optimizer with a learning rate scheduler
6. Evaluation:
   - Performance metrics: Accuracy, Precision, Recall, F1-score

## Installation
To run this project locally, follow these steps:

### Prerequisites
Ensure you have Python installed. Recommended version: **Python 3.8+**

### Clone the Repository
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Prepare the Dataset
Ensure your dataset is in the `data/` directory. If needed, update the configuration file to specify the correct paths.

### 2. Train the Model
Run the training script:
```bash
python train.py
```

### 3. Evaluate the Model
To evaluate performance on the test set:
```bash
python evaluate.py
```

### 4. Make Predictions
Use a pre-trained model to predict emotions from new audio samples:
```bash
python predict.py --file path_to_audio.wav
```

## Results
The trained CNN model achieved high accuracy in classifying speech emotions. Detailed evaluation metrics and visualization results are available in the `results/` directory.

## Future Improvements
- Expand dataset to include more emotions
- Implement attention mechanisms for better feature extraction
- Deploy the model as a web API for real-time emotion recognition

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to AIMS Cameroon 2024/2025 year group for providing the dataset and resources for this research.

