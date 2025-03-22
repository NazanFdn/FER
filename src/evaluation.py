import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model_path,
                   data_dir,
                   class_names=None,
                   img_size=(128, 128),
                   batch_size=16,
                   color_mode='grayscale'):

    # 1) Load the model
    model = tf.keras.models.load_model(model_path, compile=True)
    print(f"[INFO] Model loaded from: {model_path}")

    # 2) Create a test ImageDataGenerator (no data augmentation)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)

    # 3) Flow the data from directory
    test_generator = test_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        color_mode=color_mode,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False  # keep order
    )

    # If class_names not provided, derive from generator
    if class_names is None:
        # test_generator.class_indices is a dict: {'className': index}
        # We invert it to get a sorted list of class names
        class_map = test_generator.class_indices
        # Sort by value (index) to ensure correct order
        class_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    print("[INFO] Class names:", class_names)

    # 4) Predict on the entire dataset
    # model.predict returns probabilities of shape (num_samples, num_classes)
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Ground truth labels from the generator
    y_true = test_generator.classes

    # 5) Classification Report (Precision, Recall, F1 per class)
    print("\n[INFO] Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4  # more decimal places
    ))

    # 6) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("[INFO] Confusion Matrix (numerical):\n", cm)

    # 7) Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    _plot_confusion_matrix(cm, class_names)
    plt.show()

    # 8) Optional: Overall Accuracy from Confusion Matrix
    # Usually you already have test accuracy from model.evaluate(). But if you want:
    total_correct = np.diag(cm).sum()
    total_samples = cm.sum()
    overall_acc = total_correct / total_samples
    print(f"[INFO] Computed Overall Accuracy (from CM): {overall_acc:.4f}")


def _plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Internal helper to plot a confusion matrix with color coding and labels.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array, shape (num_classes, num_classes).
    class_names : list
        List of class names in the same index order as cm rows/columns.
    title : str
        Title for the confusion matrix plot.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Print number in each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_val = cm[i, j]
            plt.text(j, i, format(cell_val, 'd'),
                     horizontalalignment="center",
                     color="white" if cell_val > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose you have a "model_resnet34.keras" saved from training,
    # and a "data/test" directory with subfolders for each class.
    model_path = "model_resnet34.keras"
    test_directory = "/path/to/data/test"

    # Optionally define class names if you want them in a specific order:
    # e.g. class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    # If you omit it, they'll be auto-extracted from test_generator.class_indices.
    evaluate_model(model_path,
                   data_dir=test_directory,
                   class_names=None,
                   img_size=(128, 128),
                   batch_size=16,
                   color_mode='grayscale')
