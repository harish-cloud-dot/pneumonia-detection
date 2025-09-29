import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model("pneumonia_model.h5")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle uploaded image
        img_file = request.files["file"]
        if img_file:
            img_path = "temp.jpg"
            img_file.save(img_path)
            
            # Preprocess image
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Predict
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.5:
                result = "Pneumonia"
            else:
                result = "Normal"
            
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
