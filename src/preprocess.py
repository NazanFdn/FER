import cv2
import numpy as np

def preprocess_image(face_roi, target_size=(64, 64)):
    """
    Preprocess the detected face for model input.

    Parameters:
    - face_roi: The region of interest (ROI) containing the detected face.
    - target_size: The target size for resizing the face image.

    Returns:
    - preprocessed_image: The preprocessed image ready for the model.
    """
    # Resize to match model input size (64x64)
    face_resized = cv2.resize(face_roi, target_size)

    # Convert to grayscale (if needed)
    if len(face_resized.shape) == 3:  # Check if the image has multiple channels (i.e., RGB)
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    # Normalize & reshape for model input
    face_resized = face_resized.astype("float32") / 255.0

    # Ensure the image is in the shape (64, 64, 1) for the model
    face_resized = np.expand_dims(face_resized, axis=-1)   # (64, 64, 1)

    # Add batch dimension (1, 64, 64, 1)
    face_resized = np.expand_dims(face_resized, axis=0)

    return face_resized
