import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- Deep Learning Concept: Convolutional Neural Network (CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

# --- Streamlit Interactive Interface ---
def main():
    st.title("🔢 Interactive Digit Classifier")
    st.write("Upload a handwritten digit (28x28) to see the CNN in action!")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L') # Convert to Grayscale
        st.image(image, caption='Uploaded Image', width=150)
        
        # Preprocessing to match MNIST training
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(image).unsqueeze(0)

        # Load model (Mocking the weights for demonstration)
        model = SimpleCNN()
        model.eval()

        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1, keepdim=True).item()
            probabilities = F.softmax(output, dim=1).numpy()[0]

        st.success(f"Prediction: {prediction}")
        st.bar_chart(probabilities)

if __name__ == "__main__":
    main()
