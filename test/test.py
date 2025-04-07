"""
inference.py

Runs inference on a test directory with 7 subfolders:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
Each subfolder contains images for that emotion class.
Stores the ground-truth and predicted labels in an .npz file
for quick evaluation later.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(face_bgr):
    """
    Convert the BGR image to grayscale, resize to (48,48), scale to [0,1],
    then expand dims to (1,48,48,1) for model input.
    """
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)  # shape (48,48,1)
    face_expanded = np.expand_dims(face_expanded, axis=0)     # shape (1,48,48,1)
    return face_expanded

def main():
    model_path = "/Users/zeynep/PycharmProjects/FER/model/mobilenetV2_fer_model_mehmet.keras"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    # Define your emotion labels in the same order used by your model
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    label_to_index = {
        "Angry": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happy": 3,
        "Sad": 4,
        "Surprise": 5,
        "Neutral": 6
    }

    test_dir = "/Users/zeynep/PycharmProjects/FER/data/test"
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test directory '{test_dir}' not found.")
        return

    # We'll collect ground truth (y_true) and predictions (y_pred)
    y_true = []
    y_pred = []

    # Loop over each emotion subfolder
    for folder_name in emotion_labels:
        subfolder_path = os.path.join(test_dir, folder_name)
        if not os.path.exists(subfolder_path):
            print(f"[WARNING] Subfolder not found: {subfolder_path}")
            continue

        class_index = label_to_index[folder_name]

        for fname in os.listdir(subfolder_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(subfolder_path, fname)
                bgr_img = cv2.imread(img_path)
                if bgr_img is None:
                    print(f"[WARNING] Unable to read file: {img_path}")
                    continue

                # Preprocess and predict
                x_input = preprocess_image(bgr_img)
                preds = model.predict(x_input)  # shape: (1,7)
                predicted_class = np.argmax(preds[0])

                y_true.append(class_index)
                y_pred.append(predicted_class)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        print("[ERROR] No test images processed. Check your paths/folders.")
        return

    # Save the inference results
    output_file = "inference_results.npz"
    np.savez(output_file, y_true=y_true, y_pred=y_pred)
    print(f"[INFO] Inference complete. Saved results to '{output_file}'")

if __name__ == "__main__":
    main()
