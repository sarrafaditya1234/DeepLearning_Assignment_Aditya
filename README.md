# 🔢 Interactive Digit Recognizer

An interactive web application built with **Streamlit** and **PyTorch** that uses a **Convolutional Neural Network (CNN)** to classify handwritten digits.

## 🚀 Features
- **Interactive UI**: Upload images directly to the browser.
- **Deep Learning**: Uses a custom CNN architecture trained on the MNIST dataset.
- **Visual Feedback**: View probability distributions for each digit class.

## 🛠️ Installation
1. Clone the repo: `git clone https://github.com/yourusername/digit-recognizer.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 🧠 Model Architecture
The model consists of two convolutional layers followed by dropout and fully connected layers, optimizing for high accuracy on grayscale spatial data.
