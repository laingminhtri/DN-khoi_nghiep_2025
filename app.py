import os
import gdown
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# ======= TẢI MODEL TỪ GOOGLE DRIVE NẾU CHƯA CÓ =======
MODEL_PATH = "best_weights_model.keras"
FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # <-- Thay bằng file ID thật của bạn

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    print("Download completed.")

# ======= KHỞI TẠO FLASK APP & LOAD MODEL =======
app = Flask(__name__)
model = load_model(MODEL_PATH)

# Load model (upload model.keras lên Render project cùng source code luôn)
model = load_model("model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file.stream).resize((224, 224))  # Resize tùy kiến trúc model
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    prediction = model.predict(img_array)
    result = "nodule" if prediction[0][0] > 0.5 else "non-nodule"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
