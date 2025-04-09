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

## Dataset

The dataset consists of 48×48 pixel grayscale images of faces. The faces have been automatically registered so that they are roughly centered and occupy a similar area in each image. The task is to classify each face into one of seven emotion categories:
- 0 = Angry
- 1 = Disgust
- 2 = Fear
- 3 = Happy
- 4 = Sad
- 5 = Surprise
- 6 = Neutral

The training set consists of 28,709 examples, and the public test set consists of 3,589 examples.

**Note:** The actual dataset is not uploaded to GitHub due to its large size.

**Dataset License and Disclaimer:**  
The dataset is provided under the **Database Contents License (DbCL) v1.0**.  
- **Disclaimer:**  
  Open Data Commons is not a law firm and does not provide legal services. This document does not create an agent-client relationship. Please consult a qualified legal professional before using the dataset. The content is provided "as is" without warranties, and the Licensor disclaims any liability for damages resulting from its use.

For further details, please refer to the full text of the DbCL v1.0.

---

## Project Structure

- **FER/**
  - **data/**
    - **train/**  
      Training images organized into subfolders by emotion (e.g., "0=Angry", "1=Disgust", etc.)
    - **test/**  
      Test images organized into the 7 emotion categories
  - **model/**
    - **best_model.pth**  
      *(Not included in this repository due to its size exceeding 45 MB)*  
    - **masking.py**  
      Contains the definitions for MaskModule and MaskedResNet
    - **maskingTraining.py**  
      Training script for MaskedResNet
    - **mobilenet.py**  
      *(Optional)* Alternative model based on MobileNet
    - **oldMaskingTraining.py**  
      *(Optional)* Previous version of the training script
    - **resnet.py**  
      *(Optional)* Custom ResNet implementation
    - **resnet35.py**  
      *(Optional)* Custom ResNet variant
    - **updated_resnet/**  
      Directory containing updated ResNet implementations
  - **snapshots/**  
    Folder for storing snapshots and alerts captured during inference
  - **src/**
    - **gui.py**  
      Main GUI application for the Border Control FER System
    - **opencv.py**  
      Contains OpenCV-based face detection helper functions
  - **test/**
    - **classification_report.txt**  
      Generated classification report from testing model predictions
    - **evaluate_masked.py**  
      Script to evaluate the MaskedResNet model on the test set
    - **test.py**  
      Unit tests for model components and preprocessing functions
    - **test_results.npz**  
      (Optional) Saved test evaluation results
  - **README.md**  
    This documentation file
  - **main.py**  
    Entry point for launching the GUI (invokes `launch_gui`)

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





---

## License
**Database Contents License (DbCL) v1.0**

**The dataset used in this project is provided under the Database Contents License (DbCL) v1.0. Please note:**

Open Data Commons is not a law firm and does not provide legal services. This document does not create an agent-client relationship. Please consult a qualified legal professional licensed in your jurisdiction for advice before using this dataset. The dataset is provided “as is” without warranties, and the Licensor disclaims any liability for damages resulting from its use.