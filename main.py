from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_model
from app.predictor import run_prediction
from app.gee_fetcher import get_latest_sar_image
from app.utils import prediction_to_geojson
from PIL import Image
import requests
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/")
def root():
    return {"message": "Landslide Monitoring API running."}

@app.post("/predict")
async def predict_landslide(request: Request):
    data = await request.json()
    region = {
        "lat": data.get("lat"),
        "lon": data.get("lon"),
        "date": data.get("date")
    }
    image_url = get_latest_sar_image(region)
    response = requests.get(image_url)
    sar_image = Image.open(BytesIO(response.content)).convert("L")
    prediction = run_prediction(model, sar_image)
    geojson = prediction_to_geojson(prediction)
    return geojson

