from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Update CORS policy to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Update model loading to use relative paths
base_path = os.path.dirname(os.path.abspath(__file__))
models = {
    "potato": tf.keras.models.load_model(os.path.join(base_path, "..", "models", "poato.keras")),
    "corn": tf.keras.models.load_model(os.path.join(base_path, "..", "models", "3.keras")),
    "tomato": tf.keras.models.load_model(os.path.join(base_path, "..", "models", "4.keras")),
}

class_names = {
    "potato": ["Early Blight", "Late Blight", "Healthy"],
    "corn": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy","Corn___Leaf_Blight"],
    "tomato": ["Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot","Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus"]
}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hello, I am ready!"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    type: str = Query(..., description="Type of plant (potato, corn, or tomato)")
):
    if type not in models:
        return {"error": "Invalid type specified"}

    model = models[type]
    class_names_for_type = class_names[type]

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = model.predict(img_batch)

    predicted_class = class_names_for_type[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    uvicorn.run(app, host="0.0.0.0", port=port)
