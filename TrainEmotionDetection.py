import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data generators for loading and augmenting images
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Path to your dataset
train_data_dir = r'dataset/train'
validation_data_dir = r'dataset/test'
img_width, img_height = 48, 48  # Image dimensions
input_shape = (img_width, img_height, 1)  # Shape of input image (48x48 pixels, 1 channel for grayscale)

# Loading training data
train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    color_mode='grayscale',  # For grayscale images
    class_mode='categorical')

# Loading validation data
validation_generator = validation_data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    color_mode='grayscale',  # For grayscale images
    class_mode='categorical')

# Create the model
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Flatten())
model1.add(Dense(1024, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(7, activation='softmax'))  # Assuming 7 emotions to detect

# Compile the model
model1.compile(loss='categorical_crossentropy',
               optimizer=Adam(learning_rate=0.0001),
               metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model1.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1,
    callbacks=[early_stopping]
)

# Save the model architecture to a JSON file
model_json = model1.to_json()
with open('model2.json', 'w', encoding='utf-8') as json_file:
    json_file.write(model_json)

# Save the trained model weights
model1.save_weights('model2.weights.h5')

print("Model training complete and saved to disk.")
