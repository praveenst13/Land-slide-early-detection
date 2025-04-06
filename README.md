#  Landslide-early-detectio

## Overview 
   This project presents a real-time landslide detection system using SAR (Synthetic Aperture Radar) satellite imagery, leveraging deep learning-based segmentation. The model combines EfficientNet-B0 (CNN) and Swin Transformer to extract both spatial and contextual features from pre- and post-event SAR images. For implementation, Google Colab is used for model training and inference, while live SAR data is accessed through Google Earth Engine (GEE). The system is designed for early warning and continuous monitoring of landslide-prone regions, offering a scalable and practical solution for disaster management and mitigation.


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
   Extracts hierarchical spatial features from the input.
   Captures local and texture-level features effectively.

#### Encoder 2: Swin Transformer (Transformer)

   Extracts global contextual features using attention mechanisms.
   Handles long-range dependencies and improves semantic understanding.

#### Feature Fusion:
   The final feature maps from EfficientNet and Swin are resized and concatenated.
   A 1x1 Conv2D projection layer reduces dimensionality.

#### Decoder Head:
   3x3 Conv → ReLU → 1x1 Conv for segmentation mask generation.
   Final output is upsampled to 224×224 resolution using bilinear interpolation.

#### Final Output:
   1-channel segmentation mask highlighting landslide-affected regions.

##  Model Pipeline
   The model pipeline begins with Sentinel-1 SAR data collected using Google Earth Engine (GEE) for specific landslide-prone regions. For each event, pre-event and post-event SAR .tif images are stacked to form a 2-channel input, and corresponding ground truth masks are used for supervision. These images are resized to 224×224 and split into training, validation, and test sets. All preprocessing, visualization, and model training are performed using Google Colab, which offers a cloud-based and GPU-supported environment for efficient experimentation.

   The segmentation model uses a hybrid architecture combining EfficientNet-B0 (CNN) and Swin Transformer (Vision Transformer) to learn both local texture and global context from SAR data. Features from both backbones are fused and passed through convolutional layers to predict landslide-affected areas as a binary mask. The pipeline is designed for real-time integration, making it possible to simulate early landslide detection from live SAR updates (e.g., daily/weekly data from GEE), providing timely alerts for disaster response.



## Key Features
   This project introduces a real-time landslide segmentation framework using Sentinel-1 SAR data by leveraging advanced deep learning techniques. The core of the system is built on a hybrid model architecture that combines EfficientNet-B0 and Swin Transformer. EfficientNet-B0 is used for extracting rich spatial features efficiently, while Swin Transformer brings in the capability to capture long-range dependencies and contextual understanding. This fusion of convolutional and transformer-based features allows the model to make precise and context-aware predictions.

   The system processes dual-channel input, representing pre- and post-event SAR images, to detect terrain changes indicative of landslides. It outputs single-channel segmentation masks of size 224×224, identifying landslide-affected regions. Feature fusion is achieved using a Transformer-Conv architecture, which combines the best of both local and global feature learning.

   Performance is evaluated using Intersection over Union (IoU), and training is guided by a combination of Dice Loss, Focal Loss, and Supervised Contrastive Loss (SCL). The pipeline includes a robust dataset management system with training, validation, and test splits, enhanced by data augmentation techniques to improve generalization.

   For efficiency, mixed precision training (AMP) is utilized to accelerate the training process and reduce memory consumption. The model also supports early stopping, learning rate scheduling, and automatic saving of the best-performing model based on validation loss.

   The system is designed with live monitoring in mind and is capable of receiving and processing SAR image streams in near real-time. It is compatible with integration into platforms such as Google Earth Engine, making it suitable for operational deployment in landslide-prone areas. This combination of efficiency, accuracy, and real-time capability makes the framework a powerful tool for landslide risk mitigation.
##  Results
   The model achieved high accuracy in segmenting landslide-prone areas from SAR images using combined CNN and Transformer features.It demonstrated robust performance across multiple geographic regions with diverse terrain.The fused architecture improved generalization and early detection capability.
## Usage
stil
## License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with proper attribution.
