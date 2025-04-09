"""
Training of the Masked ResNet model
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from masking import MaskedResNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data directory with subfolders: 0=Angry, 1=Disgust, etc.
    data_dir = "../../../../../Volumes/DiskC/face_masked_resnet/data/train"  # Must be structured for ImageFolder

    # ------------------------------------------------------------------------------------------------------------------
    #  Define Data Transforms
    # ------------------------------------------------------------------------------------------------------------------
    # Since the images are already 48x48, we don't need to resize them.
    # transformations:
    #   - Convert images to grayscale and replicate to three channels (as ResNet expects 3-channel inputs)
    #   - Randomly rotate and horizontally flip for data augmentation.
    #   - Convert images to tensors. - Suitable for CPU computation
    #   - Normalize using mean and standard deviation used in ImageNet-trained ResNet models.
    train_transforms = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomRotation(25),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # Normalization for ResNet
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=data_dir, transform=train_transforms)

    # Split dataset into train/val
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 4) WeightedRandomSampler for class imbalance
    # Extract labels from train_subset
    train_labels = [dataset.samples[i][1] for i in train_subset.indices]
    classes = np.unique(train_labels)

    # Fix: Use class_weight='balanced' (not class_name='balanced')
    class_w = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_w_dict = {cls: w for cls, w in zip(classes, class_w)}
    print("Class Weights:", class_w_dict)

    # WeightedRandomSampler
    sample_weights = [class_w_dict[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # 5) DataLoaders
    # was 32, 256 resulted higher accuracy
    #batch_size = 32
    batch_size = 256
    # Number of parallel data loading workers was increased to 4.
    #workers=2
    workers=4
    # Create DataLoader for training data using the weighted sampler.
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    # Create DataLoader for validation data
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # ------------------------------------------------------------------------------------------------------------------
    # Build the Model
    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the MaskedResNet model.
    #   - selects a ResNet-18 backbone.
    #   - 'pretrained=True' loads ImageNet weights.
    #   - 'num_classes=len(dataset.classes)' ensures the output layer size matches the number of emotion classes.
    #   - 'dropout_p=0.3' applies dropout in the final classifier to prevent overfitting.
    model = MaskedResNet(arch="resnet18", pretrained=True, num_classes=len(dataset.classes), dropout_p=0.3)
    model.to(device)

    # Loss & Optimizer
    # Use cross-entropy loss for multi-class classification.
    criterion = nn.CrossEntropyLoss()  # WeightedRandomSampler => no need class_weight here
    # Used AdamW optimizer, which is a variant of Adam with decoupled weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training Loop
    # epoches increased to 100 as it resulted higher accuracy.
    #epochs = 25
    epochs = 100
    best_val_acc = 0.0 # set to track the best validation accuracy.
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            # Move inputs and labels to the selected device
            images, labels = images.to(device), labels.to(device)
            # Clear gradients of all model parameters.
            optimizer.zero_grad()
            # Forward pass: compute model predictions.
            outputs = model(images)
            # Compute loss between predicted outputs and ground truth labels.
            loss = criterion(outputs, labels)
            # Backward pass: compute gradients.
            loss.backward()
            # Update model parameters.
            optimizer.step()

            # Accumulate total loss and correct predictions for overall metrics.
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Evaluate validation xet
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        # Disable gradient calculation for validation
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # Calculate average validation loss and accuracy.
        val_loss /= val_total
        val_acc = val_correct / val_total

        # Printing each epoch's training and validation metrics.
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Select and save the best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       "../../../../../Volumes/DiskC/face_masked_resnet/best_masked_resnet18_48x48.pth")
            # Note: The model is saved on a specific computer to facilitate faster training and remote evaluation.

    print(f"Training done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
