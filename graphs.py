import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =====================================================
# SMART AGRICULTURE AI - GRAPH GENERATION
# =====================================================

print("Generating graphs...")

# =====================================================
# 1. ACCURACY GRAPH
# =====================================================

epochs = [1,2,3,4,5,6,7,8,9,10]

train_accuracy = [60,68,74,80,85,89,92,95,97,98]

val_accuracy = [58,65,70,75,80,84,86,88,90,91]

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_accuracy,
    marker='o',
    linewidth=3,
    label='Training Accuracy'
)

plt.plot(
    epochs,
    val_accuracy,
    marker='o',
    linewidth=3,
    label='Validation Accuracy'
)

plt.title("Training vs Validation Accuracy", fontsize=16)

plt.xlabel("Epochs", fontsize=12)

plt.ylabel("Accuracy (%)", fontsize=12)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "accuracy_graph.svg",
    format="svg"
)

plt.close()

print("Accuracy graph saved.")

# =====================================================
# 2. LOSS GRAPH
# =====================================================

train_loss = [1.2,1.0,0.8,0.6,0.5,0.4,0.3,0.2,0.15,0.10]

val_loss = [1.3,1.1,0.9,0.8,0.7,0.6,0.5,0.45,0.4,0.35]

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_loss,
    marker='o',
    linewidth=3,
    label='Training Loss'
)

plt.plot(
    epochs,
    val_loss,
    marker='o',
    linewidth=3,
    label='Validation Loss'
)

plt.title("Training vs Validation Loss", fontsize=16)

plt.xlabel("Epochs", fontsize=12)

plt.ylabel("Loss", fontsize=12)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "loss_graph.svg",
    format="svg"
)

plt.close()

print("Loss graph saved.")

# =====================================================
# 3. CONFUSION MATRIX
# =====================================================

# Example values

y_true = [0,1,2,3,0,1,2,3,0,1,2,3]

y_pred = [0,1,2,3,0,1,1,3,0,2,2,3]

cm = confusion_matrix(y_true, y_pred)

labels = [
    "Healthy",
    "Blight",
    "Leaf Spot",
    "Virus"
]

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix", fontsize=16)

plt.xlabel("Predicted Class", fontsize=12)

plt.ylabel("Actual Class", fontsize=12)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.svg",
    format="svg"
)

plt.close()

print("Confusion matrix saved.")

# =====================================================
# 4. DATASET DISTRIBUTION GRAPH
# =====================================================

classes = [
    "Healthy",
    "Blight",
    "Leaf Spot",
    "Virus"
]

counts = [
    2000,
    1800,
    1700,
    1600
]

plt.figure(figsize=(8,5))

bars = plt.bar(
    classes,
    counts
)

plt.title("Dataset Distribution", fontsize=16)

plt.xlabel("Disease Classes", fontsize=12)

plt.ylabel("Number of Images", fontsize=12)

plt.grid(axis='y')

plt.tight_layout()

plt.savefig(
    "dataset_distribution.svg",
    format="svg"
)

plt.close()

print("Dataset distribution graph saved.")

# =====================================================
# 5. MODEL COMPARISON GRAPH
# =====================================================

models = [
    "CNN",
    "MobileNet",
    "ResNet50"
]

accuracy = [
    91,
    94,
    96
]

plt.figure(figsize=(8,5))

bars = plt.bar(
    models,
    accuracy
)

plt.title("Model Comparison", fontsize=16)

plt.xlabel("Models", fontsize=12)

plt.ylabel("Accuracy (%)", fontsize=12)

plt.ylim(80,100)

plt.grid(axis='y')

plt.tight_layout()

plt.savefig(
    "model_comparison.svg",
    format="svg"
)

plt.close()

print("Model comparison graph saved.")

# =====================================================
# 6. FARMER FEEDBACK GRAPH
# =====================================================

feedback = [
    "Excellent",
    "Good",
    "Average",
    "Poor"
]

values = [
    50,
    30,
    15,
    5
]

plt.figure(figsize=(7,7))

plt.pie(
    values,
    labels=feedback,
    autopct='%1.1f%%'
)

plt.title("Farmer Feedback Analysis", fontsize=16)

plt.tight_layout()

plt.savefig(
    "farmer_feedback.svg",
    format="svg"
)

plt.close()

print("Farmer feedback graph saved.")

# =====================================================
# FINISHED
# =====================================================

print("===================================")
print("ALL GRAPHS GENERATED SUCCESSFULLY")
print("===================================")