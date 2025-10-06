# utils.py
import base64, io, os
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms

MEAN = (0.1307,)
STD = (0.3081,)

def base64_to_pil(b64_string):
    # accepts "data:image/png;base64,...."
    header_removed = b64_string.split(",")[-1]
    data = base64.b64decode(header_removed)
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_input_tensor(pil_img):
    """
    Convert a PIL RGB or L image from the canvas to a 1x1x28x28 tensor that matches
    training normalization and orientation.
    Steps:
      - convert to grayscale
      - invert (UI draws black on white -> invert to white-on-black)
      - resize to 28x28
      - ToTensor + Normalize with MEAN/STD
    """
    img = pil_img.convert("L")
    # ALWAYS invert because our canvas has black strokes on white background
    img = ImageOps.invert(img)
    # Resize to 28x28 - FIX IS HERE
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    tensor = transform(img)  # 1x28x28
    return tensor.unsqueeze(0)  # add batch dim -> 1x1x28x28

def save_user_sample(pil_img, label_index, folder="user_data"):
    os.makedirs(folder, exist_ok=True)
    idx = len(os.listdir(folder))
    path = os.path.join(folder, f"{idx}_{label_index}.png")
    # Save as grayscale 28x28 (same preprocessing visually) - FIX IS HERE
    im = pil_img.convert("L").resize((28,28), Image.Resampling.LANCZOS)
    im.save(path)
    return path