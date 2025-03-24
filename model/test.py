import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set the directory for the test data
test_dir = "/data/test"

# Define Image Size for Resizing
img_size = (128, 128)  # Use the same image size as during training

# Create a test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the test data from the directory
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',  # Grayscale if the images are grayscale
    class_mode='categorical',  # Since we have multiple classes
    batch_size=1,  # Batch size of 1 for evaluation
    shuffle=False  # Don't shuffle for evaluation to ensure we get the order of images
)

# Load the trained model
model = load_model("model.keras")  # Replace with your model path if necessary

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Make predictions on the test data
predictions = model.predict(test_data, verbose=1)

# Map predictions to emotion labels
emotion_labels = list(test_data.class_indices.keys())

# Iterate through the test set and print out the predictions and true labels
for i, (img, label) in enumerate(zip(test_data, test_data.labels)):
    predicted_class = np.argmax(predictions[i])  # Get the class with the highest probability
    true_class = np.argmax(label)  # The true class from the test set label
    print(f"Image {i+1}: Predicted: {emotion_labels[predicted_class]}, True: {emotion_labels[true_class]}")
