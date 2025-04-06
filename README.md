# 🌍 Land-slide-early-detection

**Real-Time Landslide Segmentation using Sentinel-1 SAR Imagery**

> A deep learning system designed for **early detection and segmentation** of landslides using pre- and post-event Sentinel-1 SAR data. This solution integrates **EfficientNet-B0**, **Swin Transformer**, and **real-time prediction** capabilities, developed for scalable disaster response.

---

## 🧠 Model Overview

- **Dual Encoder**:  
  - `EfficientNet-B0`: Captures spatial features  
  - `Swin Transformer`: Captures contextual, long-range dependencies  

- **Fusion Module**: Transformer-Conv based merging of encoder outputs  
- **Input**: 2-channel SAR (Pre-Event + Post-Event `.tif`)  
- **Output**: 1-channel segmentation mask (224×224)  
- **Loss**: Hybrid Dice + Focal + SCL  
- **Post-Processing**: Thresholding, CRF (optional)

---

## 🗂️ Folder Structure

├── sar_pre_event_images/ │ ├── pre_.tif ├── sar_post_event_images/ │ ├── post_.tif ├── pre_mask/ │ ├── pre_mask_.tif ├── post_mask/ │ ├── post_.tif ├── best_model.pth ├── main.py ├── utils/ │ ├── dataset.py │ ├── model.py │ └── train.py

---

## 🚀 Features

- 🔁 Pre+Post Event Image Fusion  
- 🧩 Landslide Mask Prediction  
- 🧠 EfficientNet + Swin Transformer Hybrid  
- 📈 IoU Metric and Loss Logging  
- 🧪 Train/Val/Test Split with Augmentations  
- ⚡ Mixed Precision Training (AMP)  
- 📉 Early Stopping + LR Scheduler  
- 🔴 Live Monitoring-ready architecture  

---

## 📊 Live Monitoring Setup (Simulated)

You can simulate real-time monitoring using SAR data fed at intervals:

```python
# simulate_stream.py
import time
from predict import predict_single_image

SAR_IMAGES = ["path/to/incoming/sar1.tif", "sar2.tif", ...]

for image_path in SAR_IMAGES:
    predict_single_image(image_path)
    time.sleep(60)  # simulate 1-min interval
```
##   Requirements
Python ≥ 3.8

PyTorch ≥ 1.12

timm

rasterio

torchvision

scikit-learn

numpy

matplotlib


pip install -r requirements.txt
## 🏋️ Training

python train.py
Training includes:

Mixed precision

IoU evaluation

Saving best model (best_model.pth)

Early stopping on validation loss

## 📈 Evaluation

python evaluate.py
## Outputs:

Validation Loss

IoU Score

Sample Predictions

## 👥 Team Members
Name	            Role
Praveen	            SAR Preprocessing, Lead Developer
Madhavan	        Data Engineering, Testing
Baskaran	        Model Integration, Evaluation
Harish Ragavendra	Visualization & Live Monitoring
## Sample Results
Pre-Event	Post-Event	Predicted Mask
## Future Enhancements
Integrate with Google Earth Engine for real-time SAR feed

Add anomaly suppression and CRF post-processing

Multi-temporal data support for early warning system
