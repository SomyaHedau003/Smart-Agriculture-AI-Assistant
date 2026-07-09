from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# =====================================================
# SETTINGS
# =====================================================

img_size = 128
batch_size = 32
epochs = 10

# =====================================================
# DATA GENERATOR
# =====================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =====================================================
# DEBUG PRINTS
# =====================================================

print("===================================")
print("Classes:", train_data.class_indices)
print("Total classes:", train_data.num_classes)
print("===================================")

# =====================================================
# CNN MODEL
# =====================================================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dense(
        train_data.num_classes,
        activation='softmax'
    )
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("disease_model.h5")

print("===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

# =====================================================
# 1. REAL ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    marker='o',
    linewidth=3,
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    marker='o',
    linewidth=3,
    label='Validation Accuracy'
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "real_accuracy_graph.svg",
    format="svg"
)

plt.close()

print("Accuracy graph generated.")

# =====================================================
# 2. REAL LOSS GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    marker='o',
    linewidth=3,
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    marker='o',
    linewidth=3,
    label='Validation Loss'
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "real_loss_graph.svg",
    format="svg"
)

plt.close()

print("Loss graph generated.")

# =====================================================
# 3. REAL CONFUSION MATRIX
# =====================================================

# Get predictions

predictions = model.predict(val_data)

# Convert predictions to class index

y_pred = np.argmax(predictions, axis=1)

# Actual classes

y_true = val_data.classes

# Create confusion matrix

cm = confusion_matrix(y_true, y_pred)

# Class labels

labels = list(train_data.class_indices.keys())

# Plot confusion matrix

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    "real_confusion_matrix.svg",
    format="svg"
)

plt.close()

print("Confusion matrix generated.")

# =====================================================
# 4. REAL DATASET DISTRIBUTION GRAPH
# =====================================================

dataset_path = "dataset"

classes = []
counts = []

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        total_images = len(os.listdir(folder_path))

        classes.append(folder)

        counts.append(total_images)

# Plot dataset distribution

plt.figure(figsize=(8,5))

plt.bar(classes, counts)

plt.title("Dataset Distribution")

plt.xlabel("Disease Classes")

plt.ylabel("Number of Images")

plt.xticks(rotation=15)

plt.grid(axis='y')

plt.tight_layout()

plt.savefig(
    "real_dataset_distribution.svg",
    format="svg"
)

plt.close()

print("Dataset distribution graph generated.")

# =====================================================
# 5. REAL PREDICTION CONFIDENCE GRAPH
# =====================================================

# Use first prediction example

sample_prediction = predictions[0]

confidence_values = sample_prediction * 100

plt.figure(figsize=(8,5))

plt.bar(labels, confidence_values)

plt.title("Prediction Confidence")

plt.xlabel("Disease Classes")

plt.ylabel("Confidence (%)")

plt.ylim(0,100)

plt.grid(axis='y')

plt.tight_layout()

plt.savefig(
    "real_prediction_confidence.svg",
    format="svg"
)

plt.close()

print("Prediction confidence graph generated.")

# =====================================================
# FINAL MESSAGE
# =====================================================

print("===================================")
print("ALL REAL IEEE GRAPHS GENERATED")
print("===================================")

print("Generated Files:")

print("1. real_accuracy_graph.svg")
print("2. real_loss_graph.svg")
print("3. real_confusion_matrix.svg")
print("4. real_dataset_distribution.svg")
print("5. real_prediction_confidence.svg")