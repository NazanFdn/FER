#face_detector_image
# Face detction for images
import cv2
import numpy as np
import os

# Update these paths to point to where you have your DNN model files
PROTOTXT_PATH = os.path.join("path", "to", "models", "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join("path", "to", "models", "res10_300x300_ssd_iter_140000.caffemodel")

def detect_face_dnn(image_bgr, confidence_threshold=0.5):
    """
    Detect faces using OpenCV's DNN-based face detector.
    Returns:
        If a face is found:
            (face_roi, x, y, w, h, image_bgr)
        If no face is found:
            None

    :param image_bgr: Input image in BGR format (as loaded by cv2).
    :param confidence_threshold: Confidence threshold for discarding weak detections.
    """

    # Load the serialized model (only needs to be done once, but for a simple script we'll do it here)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    # Get dimensions
    (h, w) = image_bgr.shape[:2]

    # Convert BGR to blob for DNN input
    blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # Initialize variables for the best detection
    best_confidence = 0.0
    best_box = None

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Discard detections below the confidence threshold
        if confidence > confidence_threshold and confidence > best_confidence:
            # Compute bounding box relative to the image size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clip coordinates to the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            best_confidence = confidence
            best_box = (startX, startY, endX, endY)

    if best_box is None:
        # No face passed the threshold
        return None

    # Extract ROI of the best face
    (startX, startY, endX, endY) = best_box
    face_roi = image_bgr[startY:endY, startX:endX]

    # Draw a bounding box around the face (optional)
    cv2.rectangle(image_bgr, (startX, startY), (endX, endY), (0, 255, 0), 2)

    x, y, w_box, h_box = startX, startY, (endX - startX), (endY - startY)
    return face_roi, x, y, w_box, h_box, image_bgr
