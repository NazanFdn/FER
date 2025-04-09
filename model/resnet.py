"""
    Baseline models
    Resnet50

"""


import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import mixed_precision
import os
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf

# Set the TensorFlow backend to use MPS (Metal Performance Shaders)
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))


# mixed precision for faster training
mixed_precision.set_global_policy("mixed_float16")

# Set dataset directory
data_dir = os.environ.get("DATA_DIR", "/Users/zeynep/PycharmProjects/FER/data/train")

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

batch_size = 8
img_size = (64, 64)

# Load Train & Validation Data from `data_dir`
train_data = train_datagen.flow_from_directory(
    data_dir, target_size=img_size, color_mode='grayscale',
    class_mode='categorical', batch_size=batch_size, shuffle=True, subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir, target_size=img_size, color_mode='grayscale',
    class_mode='categorical', batch_size=batch_size, shuffle=False, subset='validation'
)

# Load Pretrained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

# Convert grayscale to RGB using 1x1 convolution
input_layer = tf.keras.layers.Input(shape=(64, 64, 1))
rgb_layer = tf.keras.layers.Conv2D(3, (1, 1), activation=None)(input_layer)
base_output = base_model(rgb_layer)


# Add Classification Layers
x = GlobalAveragePooling2D()(base_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Use EarlyStopping and ReduceLROnPlateau
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Track Training Time
start_time = time.time()

#  Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  #  Increased epochs for better training
    steps_per_epoch=len(train_data) // 2,  # Adjusted steps per epoch
    validation_steps=len(val_data) // 2,
    callbacks=[early_stop, reduce_lr]
)

#  Calculate Training Time
end_time = time.time()
training_time = end_time - start_time

# Print Final Accuracy and Training Time
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
model.save("model.keras")
print(f"\n Training Completed in {training_time:.2f} seconds")
print(f"📌 Final Training Accuracy: {final_train_acc:.4f}")
print(f"📌 Final Validation Accuracy: {final_val_acc:.4f}")




"""




    #   UPDATED RESNET50 MODEL





import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight


print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# mixed precision for faster training
mixed_precision.set_global_policy("mixed_float16")

# Set dataset directory
data_dir = os.environ.get("DATA_DIR", "/Users/zeynep/PycharmProjects/FER/data/train")

# Define Data Augmentation with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,  # Keep the validation split
    rotation_range=25,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    horizontal_flip=True,  # Random horizontal flips
    fill_mode='nearest'  # Fill pixels after transformations
)

batch_size = 16  # Increased batch size

# Increase image size for better learning (128x128)
img_size = (128, 128)

# Load Train & Validation Data from `data_dir`
train_data = train_datagen.flow_from_directory(
    data_dir, target_size=img_size, color_mode='grayscale',
    class_mode='categorical', batch_size=batch_size, shuffle=True, subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir, target_size=img_size, color_mode='grayscale',
    class_mode='categorical', batch_size=batch_size, shuffle=False, subset='validation'
)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_data.classes), y=train_data.classes)
class_weights_dict = dict(enumerate(class_weights))

# Load Pretrained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Convert grayscale to RGB using 1x1 convolution
input_layer = tf.keras.layers.Input(shape=(128, 128, 1))  # Change input size to match new image size
rgb_layer = tf.keras.layers.Conv2D(3, (1, 1), activation=None)(input_layer)  # Convert grayscale to RGB
base_output = base_model(rgb_layer)

# Add Classification Layers
x = GlobalAveragePooling2D()(base_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Use AdamW optimizer
optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Use EarlyStopping and ReduceLROnPlateau
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Track Training Time
start_time = time.time()

# Train the Model with Class Weights and Augmentation
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,  # Train for more epochs
    batch_size=batch_size,
    class_weight=class_weights_dict,  # Apply class weights
    steps_per_epoch=len(train_data) // 2,
    validation_steps=len(val_data) // 2,
    callbacks=[early_stop, reduce_lr]
)

# Calculate Training Time
end_time = time.time()
training_time = end_time - start_time

# Print Final Accuracy and Training Time
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
model.save("updated_model.keras")
print(f"\nTraining Completed in {training_time:.2f} seconds")
print(f"📌 Final Training Accuracy: {final_train_acc:.4f}")
print(f"📌 Final Validation Accuracy: {final_val_acc:.4f}")





"""