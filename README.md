# Flood Area Segmentation using U-Net

## Overview

This project implements a semantic segmentation model to identify flooded regions in satellite images using the U-Net architecture. The model is trained to perform pixel-wise binary classification, where each pixel is classified as either flooded or non-flooded.

---

## Objective

The objective of this project is to accurately segment flooded areas from satellite imagery using a deep learning-based semantic segmentation model.

---

## Dataset

The project uses a flood area segmentation dataset containing:

- RGB satellite images
- Corresponding binary flood masks

https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

### Directory Structure

```
data/
├── image/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── mask/
    ├── image1.png
    ├── image2.png
    └── ...
```

Each image has a corresponding mask with the same filename.

---

## Data Preprocessing

The following preprocessing steps are performed:

- Images are resized to **256 × 256**
- Images are converted from **BGR to RGB**
- Pixel values are normalized to the range **[0, 1]**
- Masks are resized to **256 × 256**
- Masks are normalized to binary values (0 and 1)
- Masks are expanded to shape `(256, 256, 1)`

---

## Train-Test Split

The dataset is divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

using `train_test_split` from Scikit-learn.

---

## Model Architecture

The model follows the standard U-Net architecture consisting of:

- **Encoder**
  - Convolution layers (`Conv2D`)
  - Max Pooling (`MaxPooling2D`)
- **Bottleneck**
- **Decoder**
  - UpSampling (`UpSampling2D`)
  - Skip Connections (`Concatenate`)
- **Output Layer**
  - 1×1 Convolution (`Conv2D`)
  - Sigmoid activation

### Input Shape

```
256 × 256 × 3
```

### Output Shape

```
256 × 256 × 1
```

---

## Training Configuration

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Metric | Dice Coefficient |
| Epochs | 20 |
| Batch Size | 4 |

---

## Evaluation

The trained model is evaluated on the test dataset using:

- Test Loss
- Dice Coefficient

---
## Dice Coefficient

The Dice Coefficient is the primary evaluation metric used in this project. It measures the overlap between the predicted segmentation mask and the ground truth mask.

- **Dice = 1.0** → Perfect segmentation
- **Dice = 0.0** → No overlap

Formula:

```text
Dice = (2 × TP) / (2 × TP + FP + FN)
```

where:
- **TP** = True Positives
- **FP** = False Positives
- **FN** = False Negatives

The Dice Coefficient is widely used in semantic segmentation because it directly measures how well the predicted region matches the ground truth, especially when the segmented object occupies only a small portion of the image.
---

## Prediction

After training, the model predicts flood masks on test images.

The notebook visualizes:

- Original Image
- Ground Truth Mask
- Predicted Mask

---

## Model Saving

The trained model is saved as:

```
models/
└── flood_unet.keras
```

---

## Project Structure

```
Flood-Area-Segmentation/
│
├── data/
│   ├── image/
│   └── mask/
│
├── models/
│   └── flood_unet.keras
│
├──notebooks/
│   └── Untitled.ipynb
│
└── README.md
```

---

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn

Install the required packages using:

```bash
pip install tensorflow numpy opencv-python matplotlib pandas scikit-learn
```

---

## Results

The notebook reports:

- Training Loss
- Validation Loss
- Dice Coefficient
- Test Dice Coefficient

It also displays qualitative prediction results by comparing predicted masks with the corresponding ground truth masks.

---

## Future Improvements

Possible improvements include:

- Data augmentation
- Dice Loss or combined BCE + Dice Loss
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Attention U-Net or U-Net++
- Higher-resolution inputs

---

## Author

**Shyam Gayke**
