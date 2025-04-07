# evaluate.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_inference():
    data = np.load("inference_results.npz")
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    report = classification_report(y_true, y_pred, target_names=emotion_labels)
    print("\nClassification Report:\n", report)

    accuracy = (y_true == y_pred).mean()
    print(f"Overall Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate_inference()
