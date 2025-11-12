import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load your trained model
model = load_model("pneumonia_model.h5")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction_text="No file uploaded")

    img_file = request.files["file"]
    if img_file.filename == "":
        return render_template("index.html", prediction_text="Please select a file")

    try:
        # Save image temporarily
        img_path = "temp.jpg"
        img_file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

        # Clean up
        os.remove(img_path)

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
