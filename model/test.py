import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Paths & Hyperparameters
# -----------------------------
TEST_DIR   = os.getenv("TEST_DIR", "data/test")
MODEL_PATH = "model_resnet34.keras"
IMG_SIZE   = (128, 128)
BATCH_SIZE = 16

# -----------------------------
# 1) Load the Saved Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Flow from the test directory
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# 3) Evaluate the Model
# -----------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
