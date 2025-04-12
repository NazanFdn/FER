"""

Previous Masked ResNet model

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

# Import the MaskedResNet model
from model.masking import MaskedResNet
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Path to your training data
    data_dir = "/Users/zeynep/PycharmProjects/FER/data/train"

    train_transforms = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomRotation(15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # IMPORTANT: The same normalization used for ResNet pretraining
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root=data_dir, transform=train_transforms)
    print("Total images in dataset:", len(dataset))

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                             generator=torch.Generator().manual_seed(42))
    print(f"Train subset: {train_size} images, Val subset: {val_size} images.")

    train_labels = [dataset.samples[i][1] for i in train_subset.indices]
    classes = np.unique(train_labels)

    class_w = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_w_dict = {cls: w for cls, w in zip(classes, class_w)}
    print("Class Weights:", class_w_dict)

    sample_weights = [class_w_dict[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = 32
    num_workers = 2
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MaskedResNet(
        arch="resnet18",
        pretrained=True,
        num_classes=len(dataset.classes),
        dropout_p=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 40
    best_val_acc = 0.0
    save_path = "best_masked_resnet18_48x48.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] New best model saved at val_acc={best_val_acc:.4f}")

    print(f"Training done. Best val acc: {best_val_acc:.4f}")
    print(f"Best model weights saved to: {save_path}")


if __name__ == "__main__":
    main()






"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

from masking import MaskedResNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Data directory with subfolders: e.g. 0=Angry, 1=Disgust, etc.
    data_dir = "/Users/zeynep/PycharmProjects/FER/data/train"  # Must be structured for ImageFolder

    # 2) Transforms
    # No resizing => images are already 48x48
    # We replicate grayscale->3ch with T.Grayscale(3)
    train_transforms = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomRotation(25),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # Normalization for ResNet
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=data_dir, transform=train_transforms)

    # 3) Split dataset into train/val
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
    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 6) Build MaskedResNet
    model = MaskedResNet(arch="resnet18", pretrained=True, num_classes=len(dataset.classes), dropout_p=0.3)
    model.to(device)

    # 7) Loss & Optimizer
    criterion = nn.CrossEntropyLoss()  # WeightedRandomSampler => no need class_weight here
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 8) Training Loop
    epochs = 25
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Evaluateimport os
        # import numpy as np
        # 
        # import torch
        # import torch.nn as nn
        # import torch.optim as optim
        # import torchvision.transforms as T
        # from torchvision.datasets import ImageFolder
        # from torch.utils.data import DataLoader, WeightedRandomSampler
        # 
        # from sklearn.utils.class_weight import compute_class_weight
        # 
        # # Import the MaskedResNet model
        # from model.masking import MaskedResNet
        # 
        # # ------------- Reproducibility -------------
        # import random
        # 
        # def set_seed(seed=42):
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        # 
        # def main():
        #     # Set random seed for reproducibility
        #     set_seed(42)
        # 
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     print("Using device:", device)
        # 
        #     # Path to your training data
        #     data_dir = "/Users/zeynep/PycharmProjects/FER/data/train"
        #     # Make sure your folder structure is something like:
        #     #  train/
        #     #    0=Angry/
        #     #    1=Disgust/
        #     #    2=Fear/
        #     #    ...
        #     #    6=Neutral/
        # 
        #     # 1) Define transforms for training
        #     train_transforms = T.Compose([
        #         T.Grayscale(num_output_channels=3),
        #         # Slightly smaller rotation to avoid too much distortion
        #         T.RandomRotation(15),
        #         T.RandomHorizontalFlip(),
        #         T.ToTensor(),
        #         # IMPORTANT: The same normalization used for ResNet pretraining
        #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ])
        # 
        #     # 2) Load dataset with ImageFolder
        #     dataset = ImageFolder(root=data_dir, transform=train_transforms)
        #     print("Total images in dataset:", len(dataset))
        # 
        #     # 3) Split dataset into train & validation
        #     val_ratio = 0.2
        #     val_size = int(len(dataset) * val_ratio)
        #     train_size = len(dataset) - val_size
        #     train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size],
        #                                                              generator=torch.Generator().manual_seed(42))
        #     print(f"Train subset: {train_size} images, Val subset: {val_size} images.")
        # 
        #     # 4) Weighted sampler to handle imbalance
        #     train_labels = [dataset.samples[i][1] for i in train_subset.indices]
        #     classes = np.unique(train_labels)
        # 
        #     class_w = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
        #     class_w_dict = {cls: w for cls, w in zip(classes, class_w)}
        #     print("Class Weights:", class_w_dict)
        # 
        #     # WeightedRandomSampler
        #     sample_weights = [class_w_dict[label] for label in train_labels]
        #     sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # 
        #     # 5) DataLoaders
        #     batch_size = 32
        #     num_workers = 2
        #     train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        #     val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # 
        #     # 6) Build Model
        #     model = MaskedResNet(
        #         arch="resnet18",
        #         pretrained=True,
        #         num_classes=len(dataset.classes),  # 7 typically
        #         dropout_p=0.3
        #     ).to(device)
        # 
        #     # 7) Define Loss & Optimizer
        #     criterion = nn.CrossEntropyLoss()
        #     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # 
        #     # Optionally add a learning rate scheduler:
        #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        #     # or you could use: optim.lr_scheduler.ReduceLROnPlateau(...)
        # 
        #     # 8) Training Loop
        #     epochs = 40
        #     best_val_acc = 0.0
        #     save_path = "best_masked_resnet18_48x48.pth"
        # 
        #     for epoch in range(1, epochs + 1):
        #         # ---------- TRAIN ----------
        #         model.train()
        #         running_loss = 0.0
        #         correct = 0
        #         total = 0
        # 
        #         for images, labels in train_loader:
        #             images, labels = images.to(device), labels.to(device)
        # 
        #             optimizer.zero_grad()
        #             outputs = model(images)
        #             loss = criterion(outputs, labels)
        #             loss.backward()
        #             optimizer.step()
        # 
        #             running_loss += loss.item() * images.size(0)
        #             _, preds = torch.max(outputs, dim=1)
        #             correct += (preds == labels).sum().item()
        #             total += labels.size(0)
        # 
        #         train_loss = running_loss / total
        #         train_acc = correct / total
        # 
        #         # ---------- VALIDATE ----------
        #         model.eval()
        #         val_running_loss = 0.0
        #         val_correct = 0
        #         val_total = 0
        #         with torch.no_grad():
        #             for images, labels in val_loader:
        #                 images, labels = images.to(device), labels.to(device)
        #                 outputs = model(images)
        #                 loss = criterion(outputs, labels)
        #                 val_running_loss += loss.item() * images.size(0)
        #                 _, preds = torch.max(outputs, dim=1)
        #                 val_correct += (preds == labels).sum().item()
        #                 val_total += labels.size(0)
        # 
        #         val_loss = val_running_loss / val_total
        #         val_acc = val_correct / val_total
        # 
        #         # Step the scheduler
        #         scheduler.step()
        # 
        #         print(f"Epoch {epoch}/{epochs} | "
        #               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        #               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # 
        #         # ---------- CHECKPOINT ----------
        #         if val_acc > best_val_acc:
        #             best_val_acc = val_acc
        #             torch.save(model.state_dict(), save_path)
        #             print(f"[INFO] New best model saved at val_acc={best_val_acc:.4f}")
        # 
        #     print(f"Training done. Best val acc: {best_val_acc:.4f}")
        #     print(f"Best model weights saved to: {save_path}")
        # 
        # 
        # if __name__ == "__main__":
        #     main()
        # 
        # 
        # 
        # 
        # 
        # 
        # """
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as T
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, WeightedRandomSampler
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight
# import os
#
# from masking import MaskedResNet
#
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 1) Data directory with subfolders: e.g. 0=Angry, 1=Disgust, etc.
#     data_dir = "/Users/zeynep/PycharmProjects/FER/data/train"  # Must be structured for ImageFolder
#
#     # 2) Transforms
#     # No resizing => images are already 48x48
#     # We replicate grayscale->3ch with T.Grayscale(3)
#     train_transforms = T.Compose([
#         T.Grayscale(num_output_channels=3),
#         T.RandomRotation(25),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         # Normalization for ResNet
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     dataset = ImageFolder(root=data_dir, transform=train_transforms)
#
#     # 3) Split dataset into train/val
#     val_ratio = 0.2
#     val_size = int(len(dataset) * val_ratio)
#     train_size = len(dataset) - val_size
#     train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     # 4) WeightedRandomSampler for class imbalance
#     # Extract labels from train_subset
#     train_labels = [dataset.samples[i][1] for i in train_subset.indices]
#     classes = np.unique(train_labels)
#
#     # Fix: Use class_weight='balanced' (not class_name='balanced')
#     class_w = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
#     class_w_dict = {cls: w for cls, w in zip(classes, class_w)}
#     print("Class Weights:", class_w_dict)
#
#     # WeightedRandomSampler
#     sample_weights = [class_w_dict[label] for label in train_labels]
#     sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
#
#     # 5) DataLoaders
#     batch_size = 32
#     train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2)
#     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     # 6) Build MaskedResNet
#     model = MaskedResNet(arch="resnet18", pretrained=True, num_classes=len(dataset.classes), dropout_p=0.3)
#     model.to(device)
#
#     # 7) Loss & Optimizer
#     criterion = nn.CrossEntropyLoss()  # WeightedRandomSampler => no need class_weight here
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#
#     # 8) Training Loop
#     epochs = 25
#     best_val_acc = 0.0
#     for epoch in range(1, epochs + 1):
#         model.train()
#         train_loss, train_correct, train_total = 0.0, 0, 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * images.size(0)
#             _, preds = torch.max(outputs, 1)
#             train_correct += (preds == labels).sum().item()
#             train_total += labels.size(0)
#
#         train_loss = train_loss / train_total
#         train_acc = train_correct / train_total
#
#         # Evaluate
#         model.eval()
#         val_loss, val_correct, val_total = 0.0, 0, 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * images.size(0)
#                 _, preds = torch.max(outputs, 1)
#                 val_correct += (preds == labels).sum().item()
#                 val_total += labels.size(0)
#
#         val_loss /= val_total
#         val_acc = val_correct / val_total
#
#         print(f"Epoch {epoch}/{epochs} | "
#               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
#               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#
#         # Save best
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_masked_resnet18_48x48.pth")
#
#     print(f"Training done. Best val acc: {best_val_acc:.4f}")
#
#
# if __name__ == "__main__":
#     main()
"""
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_masked_resnet18_48x48.pth")

    print(f"Training done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
"""