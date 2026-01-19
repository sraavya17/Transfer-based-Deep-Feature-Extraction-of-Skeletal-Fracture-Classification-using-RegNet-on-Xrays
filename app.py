import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

model.load_state_dict(torch.load("best_model_4.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["Fractured", " Not Fractured"]

# Streamlit UI
st.title("Bone Fracture Detection App")

uploaded_file = st.file_uploader("Upload an X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probs).item()

    st.subheader(f"Prediction: {class_names[pred]}")
    st.write(f"Confidence: {probs[pred].item()*100:.2f}%")