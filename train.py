import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
import json
import os

# --- Settings ---
DATASET_DIR = 'dataset_10k'  # Your dataset directory
IMG_SIZE = (224, 224)        # 🚀 CHANGED: 224x224 is best for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 10                  # Number of times the model will train

print("Starting the training process...")


train_datagen = ImageDataGenerator(
    rescale=1./255,          
    rotation_range=20,       # 🚀 NEW: Rotate images slightly
    zoom_range=0.2,          # 🚀 NEW: Zoom in slightly
    horizontal_flip=True,    # 🚀 NEW: Flip images
    validation_split=0.2     # Reserve 20% of data for validation
)


valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("\nTraining Data:")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

print("\nValidation Data:")
validation_generator = valid_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_indices.json', 'w') as f:
    json.dump(labels, f)
print("Disease names successfully saved to 'class_indices.json'.")


print("Downloading MobileNetV2 Base Model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # 🚀 NEW: Prevents fake 100% confidence (Overfitting)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. Starting Model Training ---
print("\nModel is training... (This will give much higher accuracy now!)")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# --- 5. Saving the Trained Model ---
model.save('crop_model.h5')
print("\nCongratulations! Highly Accurate Model saved successfully as 'crop_model.h5'!")