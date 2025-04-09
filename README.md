
# Border Control Facial Expression Recognition System

**Author:** Nazan Kafadaroglu  
**Date:** 2025-04-01

---

## Overview

This project implements a Facial Expression Recognition (FER) system designed for border control applications. The system is a modified ResNet architecture that integrates a learnable "masking" mechanism to selectively amplify or suppress features. By doing so, the network aims to improve robustness, especially in security-critical scenarios such as border control.

The project consists of several key components:

- **Model Definition:**  
  A modified ResNet architecture (`MaskedResNet` with `MaskModule`) that applies feature masking by scaling the layer outputs via a learnable mask.

- **Training Pipeline:**  
  Scripts for training the MaskedResNet model using a dataset organized into subfolders (e.g., `0=Angry`, `1=Disgust`, etc.). The training pipeline includes data augmentation, normalization, and class imbalance handling via a weighted sampler.

- **Graphical User Interface (GUI):**  
  A Tkinter-based GUI that enables real-time facial expression detection using a webcam or video files. The GUI displays video frames, overlays predictions with confidence charts, and reports system resource usage.

- **Face Detection:**  
  A helper function based on OpenCV's Haar Cascade classifier is used to detect faces in captured frames.

- **Unit Tests:**  
  Unit tests verify that key functions in the model and preprocessing pipelines behave as expected.

---

## Project Structure


- **FER/**
  - **data/**
    - **train/** 
    - **test/**
  - **model/**
    - **best_model.pth**  
    - **masking.py**  
    - **maskingTraining.py**  
    - **mobilenet.py**  
    - **oldMaskingTraining.py**  
    - **resnet.py**  
    - **resnet35.py**  
    - **updated_resnet/**  
  - **snapshots/**  
  - **src/**
    - **gui.py**  
    - **opencv.py**  
  - **test/**
    - **classification_report.txt**  
    - **evaluate_masked.py**  
    - **test.py**  
    - **test_results.npz**  
  - **README.md**  
  - **main.py**

---


---

## Requirements

- **Python 3.8+**
- **PyTorch** and **torchvision**
- **OpenCV** (`opencv-python`)
- **NumPy**
- **scikit-learn**
- **tkinter** (usually included with Python)
- **psutil**
- **matplotlib**

Install the necessary packages via pip:

```bash
pip install torch torchvision opencv-python pillow numpy scikit-learn psutil matplotlib

```


---

## Usage


**Training the Model**
Dataset Preparation:
Place your training images in the data/train folder, with each subfolder representing an emotion class (e.g., 0=Angry, 1=Disgust, etc.). Ensure images are 48×48 pixels.

**Run the Training Script**
Execute the training script:


```bash
python path/to/your/training_script.py

```

The script performs data augmentation, normalizes images, handles class imbalance using a weighted sampler, and saves the best model (e.g., best_masked_resnet18_48x48.pth) based on validation accuracy.


```bash
python main.py
```
This launches a full-screen login window; upon successful login, the main interface appears. The system supports webcam streaming as well as video file playback for real-time emotion detection.







---

## Final Comparisons and Results
After training and evaluation, the MaskedResNet model achieved the following performance on the test set:

Overall Accuracy: 63.18%

The results demonstrate that the proposed method achieves competitive performance across all emotion classes, with particularly strong performance for the "happy" and "surprise" classes. Slight trade-offs in accuracy for some classes reflect the inherent challenge of balancing the model's robustness and overall performance.