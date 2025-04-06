import numpy as np
from torchvision import transforms
from PIL import Image
import torch

def run_prediction(model, sar_image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(sar_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.sigmoid(output).squeeze().numpy()
    return prediction
