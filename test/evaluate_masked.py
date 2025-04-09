import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model.masking import MaskedResNet

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Test data directory and model path
    test_dir = "/Users/zeynep/PycharmProjects/FER/data/test"
    weights_path = "/Users/zeynep/PycharmProjects/FER/model/best_model.pth"

    # Check if the model weights file exists
    if not os.path.exists(weights_path):
        print(f"[ERROR] No weights found at: {weights_path}")
        return

    # Transforms for testing (no random augmentations)
    test_transforms = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,  # Increased batch size for faster evaluation
        shuffle=False,
        num_workers=4
    )

    # Print the classes to ensure consistency
    print("Test dataset classes:", test_dataset.classes)

    # Load the model and weights
    model = MaskedResNet(arch="resnet18", pretrained=False, num_classes=len(test_dataset.classes), dropout_p=0.3)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model weights from {weights_path}")

    # Inference and evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            # Append predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy calculation
    test_acc = accuracy_score(all_labels, all_preds) * 100
    print(f"[TEST] Accuracy = {test_acc:.2f}%")

    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print("\nClassification Report:")
    print(class_report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, test_dataset.classes)

    # Save the results to .npz for later analysis
    np.savez("test_results.npz", accuracy=test_acc, y_true=all_labels, y_pred=all_preds)

    # Save classification report to a text file
    with open("classification_report.txt", "w") as f:
        f.write(f"Accuracy: {test_acc:.2f}%\n")
        f.write(class_report)

    print(f"[INFO] Test results saved to 'test_results.npz' and 'classification_report.txt'")

if __name__ == "__main__":
    main()
