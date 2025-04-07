import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Sklearn metrics for evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import your MaskedResNet definition
from model.masking import MaskedResNet

def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Path to your test set + saved model weights
    test_dir = "/Users/zeynep/PycharmProjects/FER/data/test"
    weights_path = "/Users/zeynep/PycharmProjects/FER/model/best_masked_resnet18_48x48.pth"

    # 3) Define the same transforms used for training, except typically
    # we skip random augmentations (like RandomRotation) for the test set.
    test_transforms = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # Show the classes (should match training)
    print("Test dataset classes:", test_dataset.classes)

    # 5) Build the same model architecture used during training
    #    e.g. MaskedResNet with resnet18, 7 classes, dropout=0.3
    model = MaskedResNet(
        arch="resnet18",
        pretrained=False,         # We do NOT load ImageNet weights here
        num_classes=len(test_dataset.classes),
        dropout_p=0.3
    )
    model.to(device)

    # 6) Load the best weights
    if not os.path.exists(weights_path):
        print(f"[ERROR] No weights found at: {weights_path}")
        return
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded model weights from {weights_path}")

    # 7) Inference / Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)       # shape: [batch_size, num_classes]

            # Option 1: predictions from argmax
            _, preds = torch.max(outputs, dim=1)

            # Save predictions + labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 8) Compute metrics
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"[TEST] Accuracy = {test_acc*100:.2f}%")

    # classification_report
    class_report = classification_report(
        all_labels, all_preds,
        target_names=test_dataset.classes
    )
    print("\nClassification Report:")
    print(class_report)

    # confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
