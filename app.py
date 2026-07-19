from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os
import matplotlib.pyplot as plt

app = Flask(__name__)

MODEL_PATH = "simple_cnn_lung_model.keras"
IMG_SIZE = 128
STATIC_DIR = "static"

SUMMARY_IMG_PATH = f"{STATIC_DIR}/model_summary.png"
ORIGINAL_IMG_PATH = f"{STATIC_DIR}/original_input.png"
PROB_CHART_PATH = f"{STATIC_DIR}/probabilities.png"

os.makedirs(STATIC_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

class_names = [
    "Adenocarcinoma",
    "Benign",
    "Large Cell Carcinoma",
    "Malignant",
    "Normal",
    "Squamous Cell Carcinoma"
]

def save_model_summary():
    if os.path.exists(SUMMARY_IMG_PATH):
        return

    import io as io_stream
    stream = io_stream.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    text = stream.getvalue()

    plt.figure(figsize=(14, text.count("\n") * 0.32))
    plt.text(0.01, 0.99, text, family="monospace", fontsize=10, va="top")
    plt.axis("off")
    plt.savefig(SUMMARY_IMG_PATH, dpi=200, bbox_inches="tight")
    plt.close()

save_model_summary()

def save_probability_chart(probabilities):
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, probabilities, color="#1abc9c")
    plt.ylabel("Probability (%)")
    plt.title("Model Confidence Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(PROB_CHART_PATH, dpi=150)
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = probabilities = None
    chart_available = False

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img.save(ORIGINAL_IMG_PATH)

            arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = model.predict(arr)[0]

            idx = int(np.argmax(preds))
            prediction = class_names[idx]
            confidence = round(float(preds[idx]) * 100, 2)
            probabilities = list(zip(class_names, (preds * 100).round(2)))

            save_probability_chart(preds * 100)
            chart_available = True

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        chart_available=chart_available,
        summary_img="model_summary.png",
        original_img="original_input.png",
        prob_chart="probabilities.png"
    )

if __name__ == "__main__":
    app.run(debug=True)
