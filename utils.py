import numpy as np
import json

def prediction_to_geojson(prediction, threshold=0.5):
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    h, w = prediction.shape
    for y in range(h):
        for x in range(w):
            if prediction[y, x] > threshold:
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [x, y]
                    },
                    "properties": {
                        "risk_score": float(prediction[y, x])
                    }
                })
    return geojson
