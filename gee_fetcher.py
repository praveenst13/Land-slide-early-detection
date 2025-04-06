import ee
import geemap
from PIL import Image
import numpy as np

ee.Initialize()

def get_latest_sar_image(region):
    point = ee.Geometry.Point(region["lon"], region["lat"])
    start_date = ee.Date(region["date"]).advance(-7, 'day')
    end_date = ee.Date(region["date"])

    collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(point) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .select("VV")

    image = collection.mean().clip(point.buffer(10000))

    url = image.getThumbURL({
        "region": point.buffer(10000),
        "dimensions": 256,
        "format": "png",
        "min": -25,
        "max": 0
    })

    return url
