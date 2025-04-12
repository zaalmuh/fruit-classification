import numpy as np
from PIL import Image

# Update this with your actual class names
CLASS_NAMES = ["fresh apple", "fresh banana", "fresh orange", "rotten apple", "rotten banana", "rotten orange"]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Change size to match your model
    image_array = np.array(image)
    # image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)
