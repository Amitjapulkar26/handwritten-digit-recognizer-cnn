# ✍️ Handwritten Digit Recognizer

A CNN-based deep learning application that recognizes handwritten digits (0–9) using the MNIST dataset.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-success" />
  <img src="https://img.shields.io/badge/Status-Deployed-brightgreen" />
</p>

## 🚀 Live Demo

**Try the application:** [Handwritten Digit Recognizer](YOUR_STREAMLIT_APP_URL)

Upload a PNG, JPG, or JPEG image containing one handwritten digit and view the predicted digit, confidence, and class probabilities.

> Replace `YOUR_STREAMLIT_APP_URL` with your actual Streamlit Community Cloud application URL.

## 📸 Project Screenshots

Add screenshots of your actual running application to a folder named `screenshots` in this repository.

<p align="center">
  <img src="screenshots/home.png" width="85%" alt="Application interface" />
  <br>
  <em>Application Interface</em>
</p>

<p align="center">
  <img src="screenshots/prediction.png" width="85%" alt="Prediction result" />
  <br>
  <em>Digit Prediction Result</em>
</p>

<p align="center">
  <img src="screenshots/performance.png" width="85%" alt="Model performance" />
  <br>
  <em>Model Performance</em>
</p>

## 🎯 Project Overview

The Handwritten Digit Recognizer is an image-classification application that identifies a single handwritten numerical digit from 0 to 9.

The application uses a **Convolutional Neural Network (CNN)** implemented with TensorFlow/Keras and trained on the MNIST dataset.

An uploaded image is preprocessed to make it suitable for the model. The model then predicts the digit, and the application displays the prediction, confidence, class probabilities, and available test metrics.

## ✨ Features

- TensorFlow/Keras CNN with convolution, pooling, dropout, and softmax classification.
- Downloads MNIST automatically during training.
- Training, validation, and held-out test evaluation.
- Upload PNG/JPG/JPEG handwritten digit images.
- Image preprocessing: grayscale conversion, contrast normalization, thresholding, cropping, resizing, and centering.
- Displays predicted digit and confidence.
- Displays class probabilities and test metrics.
- Interactive Streamlit user interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and training |
| NumPy | Numerical operations |
| Pillow (PIL) | Image processing |
| Streamlit | Interactive web application |
| MNIST | Handwritten digit dataset |

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) for image classification.

- **Convolution layers:** Extract visual patterns and features from digit images.
- **Pooling layers:** Reduce feature-map dimensions.
- **Dropout:** Helps reduce overfitting.
- **Dense layers:** Perform final classification.
- **Softmax output:** Produces probabilities for ten digit classes.

### Prediction Workflow

1. User uploads a handwritten digit image.
2. Image preprocessing is performed.
3. The processed image is passed to the trained CNN.
4. The model predicts one of ten classes.
5. The application displays the predicted digit and confidence.

```text
Uploaded Handwritten Digit
          |
          v
    Image Preprocessing
          |
          v
     Trained CNN Model
          |
          v
    Softmax Classification
          |
          v
 Predicted Digit + Confidence
     + Class Probabilities
```

## 📊 Model Performance

The model is evaluated using the held-out MNIST test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | ADD ACTUAL VALUE |
| Test Loss | ADD ACTUAL VALUE |
| Dataset | MNIST |
| Output Classes | 10 (0–9) |

**Note:** Replace the placeholders with the actual values from `models/metrics.json`. Do not use training accuracy as test accuracy.

## 📚 Dataset

The MNIST dataset is loaded using:

```python
tensorflow.keras.datasets.mnist.load_data()
```

Dataset details:

- 60,000 training images.
- 10,000 test images.
- 28 × 28 pixel grayscale images.
- Ten digit classes (0–9).

The dataset is downloaded and cached automatically by Keras during the first training run.

## ⚙️ Setup on Windows / VS Code

### 1. Clone the repository

```powershell
git clone https://github.com/Amitjapulkar26/handwritten-digit-recognizer-cnn.git
```

### 2. Open the project folder

```powershell
cd handwritten-digit-recognizer-cnn
```

Open this folder in VS Code. Select **Terminal → New Terminal**.

### 3. Create a Python 3.11 virtual environment

```powershell
py -3.11 -m venv .venv
```

### 4. Activate the environment

```powershell
.venv\Scripts\activate
```

### 5. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Train the model

The first training run downloads the MNIST dataset automatically.

```powershell
python train.py
```

After training, the following files are generated:

- `models/mnist_cnn.keras`
- `models/metrics.json`

### 7. Start the application

```powershell
streamlit run app.py
```

A browser tab opens automatically. If it does not, visit:

`http://localhost:8501`

## 📂 Project Structure

```text
01_Handwritten_Digit_Recognizer/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.txt
│
├── models/
│   ├── .gitkeep
│   ├── mnist_cnn.keras
│   └── metrics.json
│
└── screenshots/
    ├── home.png
    ├── prediction.png
    └── performance.png
```

The model and metrics files are generated after training. The screenshots folder contains the application screenshots added to GitHub.

## 🎥 Project Demonstration

For an internship or technical evaluation, demonstrate the following:

1. Open the deployed Streamlit application.
2. Upload an image containing one handwritten digit.
3. Show the predicted digit and confidence.
4. Explain the class probabilities.
5. Test another handwritten digit.
6. Briefly explain how the CNN learns patterns from MNIST.

## 🗣️ Project Explanation for Evaluation

My project is a Handwritten Digit Recognizer developed using Python, TensorFlow/Keras, and Streamlit.

I trained a Convolutional Neural Network on the MNIST dataset, which contains handwritten digit images from 0 to 9.

When a user uploads an image, the application preprocesses it by converting it to grayscale, normalizing contrast, thresholding, cropping, resizing, and centering it.

The trained CNN then predicts the digit. The Streamlit interface displays the predicted digit, confidence, class probabilities, and test metrics.

This project helped me gain practical experience in image preprocessing, CNN training, model evaluation, and deploying a machine learning application.

## ⚠️ Notes and Limitations

- The model is trained on MNIST's writing style. Unusual handwriting, multiple digits, rotated images, shadows, and complex backgrounds can reduce accuracy.
- Upload one digit per image. The app does not perform multi-digit OCR.
- The displayed confidence is the model's softmax probability, not a calibrated guarantee of correctness.
- Training time depends on CPU/GPU availability. TensorFlow may run on CPU.

## 🔮 Future Improvements

- Add a drawing canvas for direct browser input.
- Improve recognition of unusual handwriting styles.
- Extend the application to recognize multiple digits using an appropriate OCR pipeline.
- Add more detailed model evaluation visualizations.

## 👨‍💻 Developer

**Amit Japulkar**

AI & Machine Learning Engineering Student

- GitHub: [@Amitjapulkar26](https://github.com/Amitjapulkar26)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/amit-japulkar-939371393/)

---

<p align="center">
  **Built with Python, TensorFlow/Keras, CNN, and Streamlit.**
  <br>
  If you find this project useful, consider giving the repository a ⭐.
</p>
