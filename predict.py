import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

MODEL_PATH = "simple_cnn_lung_model.keras"
IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "adenocarcinoma",
    "benign",
    "large_cell_carcinoma",
    "malignant",
    "normal",
    "squamous_cell_carcinoma"
]

def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = preds[0][class_index] * 100

    result_text.set(
        f"Prediction: {class_names[class_index]}\n"
        f"Confidence: {confidence:.2f}%"
    )

    img_display = ImageTk.PhotoImage(img.resize((250, 250)))
    image_label.config(image=img_display)
    image_label.image = img_display

root = tk.Tk()
root.title("Lung Cancer Detection")
root.geometry("400x500")

title = tk.Label(root, text="Lung Cancer Image Prediction",
                 font=("Arial", 14, "bold"))
title.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

btn = tk.Button(root, text="Select Image",
                command=predict_image,
                font=("Arial", 12),
                bg="#4CAF50",
                fg="white")
btn.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text,
                        font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
