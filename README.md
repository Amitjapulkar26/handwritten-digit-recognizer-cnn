# Handwritten Digit Recognizer

A CNN-based image classification application that recognizes handwritten digits (0–9) using the MNIST dataset.

## Features

- TensorFlow/Keras CNN with convolution, pooling, dropout, and softmax classification.
- Downloads MNIST automatically during training.
- Training, validation, and held-out test evaluation.
- Upload PNG/JPG/JPEG handwritten digit images.
- Image preprocessing: grayscale, contrast normalization, thresholding, crop, resize, and centering.
- Displays predicted digit, confidence, class probabilities, and test metrics.
- Streamlit user interface.

## Tech stack

Python, TensorFlow, Keras, NumPy, Pillow, Streamlit, MNIST.

## Setup on Windows / VS Code

Open this folder in VS Code. Open Terminal → New Terminal and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the model (first run downloads MNIST):

```powershell
python train.py
```

Start the application:

```powershell
streamlit run app.py
```

A browser tab opens. If it does not, visit `http://localhost:8501`.

## Dataset

MNIST is loaded using `tensorflow.keras.datasets.mnist.load_data()`. It contains 60,000 training images and 10,000 test images, each a 28×28 grayscale image. The data is downloaded and cached automatically by Keras.

## Project structure

```text
01_Handwritten_Digit_Recognizer/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.txt
└── models/
    └── .gitkeep
```

After training, `models/mnist_cnn.keras` and `models/metrics.json` are generated.

## Notes and limitations

- The model is trained on MNIST's writing style. Unusual handwriting, multiple digits, rotated images, shadows, and complex backgrounds can reduce accuracy.
- Upload one digit per image. The app does not perform multi-digit OCR.
- The app's confidence is the model's softmax probability, not a calibrated guarantee.
- Training time depends on CPU/GPU. TensorFlow may run on CPU.

## GitHub publishing

1. Create a public repository named `handwritten-digit-recognizer-cnn`.
2. In this folder, run:

```powershell
git init
git add .
git commit -m "Build MNIST handwritten digit recognizer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/handwritten-digit-recognizer-cnn.git
git push -u origin main
```

3. Replace `YOUR-USERNAME` with your GitHub username.
4. Add screenshots of the working app and your actual test accuracy to this README.
5. Do not upload `.venv/` or the downloaded MNIST cache. The trained model is reproducible using `python train.py`.
