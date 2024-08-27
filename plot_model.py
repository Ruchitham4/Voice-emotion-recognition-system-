import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your Keras model
model_path_h5 = "E:/emotion-classification-from-audio-files-master/model/Emotion_Voice_Detection_Model.h5"
model_h5 = load_model(model_path_h5)

# Assuming you have history stored somewhere, for example:
history = {
    'loss': [0.1, 0.2, 0.3, 0.4],  # Replace with your actual training loss values
    'val_loss': [0.08, 0.18, 0.25, 0.35]  # Replace with your actual validation loss values
}

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)
plt.savefig("model_loss_plot.png")
plt.show()

print("Model loss plot saved as 'model_loss_plot.png'")

