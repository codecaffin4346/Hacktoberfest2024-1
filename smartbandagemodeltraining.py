import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_images(image_dir, img_size=(128, 128)):
    images = []
    labels = []
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(float(label) / 100.0)  # Normalize healing percentage to 0.0-1.0 range
    return np.array(images), np.array(labels)

# Define paths (adjust the paths to your dataset)
train_dir = "path_to_train_dataset"
test_dir = "path_to_test_dataset"

# Load and preprocess the images
X_train, y_train = load_images(train_dir)
X_test, y_test = load_images(test_dir)

# Normalize image data (0-255) to (0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape labels for regression (healing percentage)
y_train = np.array(y_train)
y_test = np.array(y_test)

# CNN Model definition
def build_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the convolutional output
    model.add(Flatten())

    # Dense layers for regression
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Add dropout to prevent overfitting

    # Output layer for predicting healing percentage
    model.add(Dense(1, activation='linear'))  # Output is a percentage (0.0 to 1.0)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

# Build the model
input_shape = (128, 128, 3)  # Image size 128x128 with 3 color channels (RGB)
model = build_model(input_shape)

# Display model summary
model.summary()

# Data Augmentation (Optional)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50
)

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
model.save("wound_healing_model.h5")

# Evaluate the model on test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Mean Absolute Error: {mae}")



  

