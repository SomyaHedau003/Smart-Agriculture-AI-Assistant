from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model("disease_model.h5")

# Class labels
classes = ["blight", "healthy", "leaf_spot", "virus"]

def predict_image(img_file):
    # Open image from Streamlit upload
    img = Image.open(img_file)

    # Convert to RGB (important)
    img = img.convert("RGB")

    # Resize to model input size
    img = img.resize((128, 128))

    # Convert to numpy
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Reshape for model
    img = np.reshape(img, (1, 128, 128, 3))

    # Prediction
    pred = model.predict(img)

    # Get class label
    predicted_class = classes[np.argmax(pred)]

    return predicted_class