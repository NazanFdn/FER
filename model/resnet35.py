"""

Baseline Model
ResNet35

"""

import time
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt

from classification_models.tfkeras import Classifiers

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

mixed_precision.set_global_policy("mixed_float16")

train_dir = os.getenv("TRAIN_DIR", "/Users/zeynep/PycharmProjects/FER/data/train")  # This is your single training folder

img_size       = (128, 128)
batch_size     = 16
num_epochs     = 50
learning_rate  = 0.001
weight_decay   = 1e-4

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
    subset='validation'
)

ResNet34, preprocess_input = Classifiers.get('resnet34')

input_layer = Input(shape=(img_size[0], img_size[1], 1))
rgb_layer   = Conv2D(3, (1, 1), activation=None)(input_layer)

base_model = ResNet34(
    input_shape=(img_size[0], img_size[1], 3),
    weights='imagenet',
    include_top=False
)

for layer in base_model.layers[:-10]:
    layer.trainable = False

base_output = base_model(rgb_layer)

x = GlobalAveragePooling2D()(base_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
model.summary()

optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

start_time = time.time()

history = model.fit(
    train_data,
    epochs=num_epochs,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr]
)

end_time = time.time()
training_time = end_time - start_time

model.save("model_resnet34.keras")

final_train_acc = history.history['accuracy'][-1]
final_val_acc   = history.history['val_accuracy'][-1]
print(f"\nTraining Completed in {training_time:.2f} seconds")
print(f"📌 Final Training Accuracy: {final_train_acc:.4f}")
print(f"📌 Final Validation Accuracy: {final_val_acc:.4f}")

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
