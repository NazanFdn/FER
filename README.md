
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
- **Pillow**
- **NumPy**
- **scikit-learn**
- **tkinter** (usually included with Python)
- **psutil**
- **matplotlib**

Install the necessary packages via pip:

```bash
pip install torch torchvision opencv-python pillow numpy scikit-learn psutil matplotlib




