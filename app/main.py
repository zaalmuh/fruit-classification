from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
from app.utils import preprocess_image, CLASS_NAMES

app = FastAPI()
model = load_model("model/model_fruit.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return JSONResponse({
        "predicted_class": predicted_class,
        "probabilities": prediction.tolist()[0]
    })
