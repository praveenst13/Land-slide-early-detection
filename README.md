#  Landslide-early-detection

## Overview 
   This project presents a real-time landslide detection system using SAR (Synthetic Aperture Radar) satellite imagery and a deep learning-based segmentation model. By combining EfficientNet-B0 (CNN) and Swin Transformer, the model effectively captures both spatial and contextual features from pre- and post-event SAR images. The system aims to provide early warnings and enable live monitoring of landslide-prone areas, offering a scalable solution for disaster management and mitigation.


## Problem Statement 
   Landslides pose a serious threat to life, infrastructure, and the environment, especially in hilly and high-rainfall regions. Traditional monitoring methods are either manual, delayed, or limited in coverage. There is a critical need for an automated, real-time system that can detect landslides early using remote sensing data.This project addresses the challenge of developing a deep learning-based landslide detection system that can process pre- and post-event SAR images to segment affected regions accurately, with the goal of enabling early warning and live situational awareness.


## Architecture
   This project uses a hybrid deep learning architecture that combines the strengths of both CNNs and Transformers for effective landslide segmentation from SAR images.
### Components:
#### Input:
   2-channel SAR images (Pre-event & Post-event)
   Size: 2×224×224
#### Channel Expansion Layer:
   A 1x1 Conv2D layer expands the input from 2 → 3 channels to match pretrained model requirements.

#### Encoder 1: EfficientNet-B0 (CNN)
   1)Extracts hierarchical spatial features from the input.
   2)Captures local and texture-level features effectively.

#### Encoder 2: Swin Transformer (Transformer)

   1)Extracts global contextual features using attention mechanisms.
   2)Handles long-range dependencies and improves semantic understanding.

#### Feature Fusion:
   1)The final feature maps from EfficientNet and Swin are resized and concatenated.
   2)A 1x1 Conv2D projection layer reduces dimensionality.

#### Decoder Head:
   1)3x3 Conv → ReLU → 1x1 Conv for segmentation mask generation.
   2)Final output is upsampled to 224×224 resolution using bilinear interpolation.

#### Final Output:
   1)1-channel segmentation mask highlighting landslide-affected regions.
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
      Name	                  Role
      Praveen	                SAR Preprocessing, Lead Developer
      Madhavan	              Data Engineering, Testing
      Baskaran	              Model Integration, Evaluation
      Harish Ragavendra	      Visualization & Live Monitoring
## Sample Results
Pre-Event	Post-Event	Predicted Mask
## Future Enhancements

1. 🌐 Integration with Google Earth Engine (GEE)
Why: To access real-time Sentinel-1 SAR data without downloading files manually
Benefit: Enables fully automated live monitoring of landslide-prone areas

2. 🧪 Anomaly Suppression + CRF Post-Processing
Why: Deep models may still misclassify rough terrain or shadows as landslides
What we’ll add:

Anomaly suppression: Eliminate false positives using terrain masks

CRF (Conditional Random Field): Sharpen segmentation masks by refining boundaries
Benefit: Increases prediction reliability and mask clarity

3. 🕒 Multi-Temporal Data Support for Early Warning
Why: One pair of SAR images may not be enough to predict slow-developing landslides
What we’ll add:

Use multiple time steps of SAR data to observe deformation patterns

Integrate time series modeling to predict landslide likelihood
Benefit: Transforms the system from event detection to early warning system

Bonus Plan 🚧: EfficientNet + Swin + U-Net Decoder
We plan to add a U-Net-style decoder for better upsampling and fine detail recovery:

Keeps skip connections

Enhances mask sharpness

Maintains real-time capability
