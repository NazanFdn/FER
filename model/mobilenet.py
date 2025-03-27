import time
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, Conv2D,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import AdamW

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# 1. Enable mixed precision for faster training on supported GPUs/CPUs
mixed_precision.set_global_policy("mixed_float16")

# 2. Set dataset directory
data_dir = os.environ.get("DATA_DIR", "data/train")

# 3. Define data generators with augmentations
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
img_size = (64, 64)  # For MobileNetV2, larger sizes (e.g. 96x96 or 128x128) may improve accuracy

# 4. Create training & validation generators
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='grayscale',    # Still load as grayscale
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
    subset='validation'
)

# 5. Build the MobileNetV2 base model
#    MobileNetV2 is pretrained on ImageNet => expects 3 channels
#    We'll insert a Conv2D to map 1 grayscale channel -> 3 RGB channels
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

# 6. Build a custom input layer for grayscale data
#    Then route through a 1x1 Conv to get 3 channels
input_layer = Input(shape=(64, 64, 1), name="input_grayscale")
rgb_layer = Conv2D(3, (1, 1), padding='same', activation=None, name="grayscale_to_rgb")(input_layer)
base_output = base_model(rgb_layer)

# 7. Add classification layers on top of base model
x = GlobalAveragePooling2D()(base_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(train_data.num_classes, activation='softmax', dtype='float32')(x)

# 8. Assemble the full model
model = Model(inputs=input_layer, outputs=output_layer)

# 9. Choose an optimizer and compile
#    AdamW is a good choice; consider a smaller learning rate for fine-tuning
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 10. Setup callbacks: EarlyStopping & ReduceLROnPlateau
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 11. Train the model
start_time = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    # Steps per epoch can be set to len(train_data) or a fraction if training speed is an issue
    steps_per_epoch=len(train_data),           # or e.g. len(train_data)//2
    validation_steps=len(val_data),
    callbacks=[early_stop, reduce_lr]
)

end_time = time.time()
training_time = end_time - start_time

# 12. Print final results
final_train_acc = history.history['accuracy'][-1]
final_val_acc   = history.history['val_accuracy'][-1]

model.save("mobilenetV2_fer_model.keras")

print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"📌 Final Training Accuracy: {final_train_acc:.4f}")
print(f"📌 Final Validation Accuracy: {final_val_acc:.4f}")
