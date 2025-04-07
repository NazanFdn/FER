import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model.masking import MaskedResNet  # Your custom model script

def preprocess_image(face_bgr):
    """
    Convert the BGR image to grayscale, resize to (48,48), normalize,
    and prepare for model input as (1,3,48,48).
    """
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_normalized = face_resized / 255.0
    face_normalized = np.repeat(face_normalized[:, :, np.newaxis], 3, axis=2)  # Convert to 3 channels
    face_tensor = T.ToTensor()(face_normalized).type(torch.float32)  # Ensure float32
    face_tensor = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(face_tensor)
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    return face_tensor

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/Users/zeynep/PycharmProjects/FER/model/best_masked_resnet18_48x48.pth"

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    # Load the trained model
    model = MaskedResNet(arch="resnet18", pretrained=False, num_classes=7, dropout_p=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.evaAysefatmahayriye123
    l()
    print(f"[INFO] Loaded model from {model_path}")

    # Emotion labels
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    label_to_index = {label: idx for idx, label in enumerate(emotion_labels)}

    # Test data directory
    test_dir = "/Users/zeynep/PycharmProjects/FER/data/test"
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test directory '{test_dir}' not found.")
        return

    # Initialize true and predicted label lists
    y_true = []
    y_pred = []

    # Perform testing
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

                # Preprocess the image and predict
                x_input = preprocess_image(bgr_img).to(device)
                with torch.no_grad():
                    outputs = model(x_input)
                    predicted_class = torch.argmax(outputs, dim=1).item()

                y_true.append(class_index)
                y_pred.append(predicted_class)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for empty results
    if len(y_true) == 0:
        print("[ERROR] No test images processed. Check your paths/folders.")
        return

    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"[INFO] Test Accuracy: {accuracy:.2f}%")

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=emotion_labels)
    print("\nClassification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, emotion_labels)

    # Save results to .npz file
    np.savez("evaluation_results.npz", y_true=y_true, y_pred=y_pred, accuracy=accuracy, report=report)
    print(f"[INFO] Evaluation results saved to 'evaluation_results.npz'")

if __name__ == "__main__":
    main()
