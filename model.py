import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Dataset directories
train_dir = r"C:\Users\Harishkumar Ganesh\OneDrive\Desktop\pneumonia-detection\chest_xray\train"
val_dir   = r"C:\Users\Harishkumar Ganesh\OneDrive\Desktop\pneumonia-detection\chest_xray\val"

# Image parameters
img_size = 150
batch_size = 32

# Data preprocessing + augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Save trained model
os.makedirs("saved_models", exist_ok=True)
model_path_h5 = os.path.join("saved_models", "pneumonia_model.h5")
model_path_keras = os.path.join("saved_models", "pneumonia_model.keras")
model.save(model_path_h5)
model.save(model_path_keras)
print(f"Model saved at:\n{model_path_h5}\n{model_path_keras}")

# Save accuracy and loss plots
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8,6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join("plots", "accuracy.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join("plots", "loss.png"))
plt.close()

print("Plots saved in 'plots' folder. Training complete!")
