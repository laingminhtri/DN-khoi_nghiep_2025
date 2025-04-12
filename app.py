import os
import gdown
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from tensorflow import keras

# ======= CẤU HÌNH GPU HOẶC CPU ĐỂ TRÁNH LỖI OOM =======
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass

# ======= TẢI MODEL TỪ GOOGLE DRIVE NẾU CHƯA CÓ =======
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/best_weights_model.keras"  # Tên file đúng
FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # <-- Thay bằng file ID thật của bạn

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    print("Download completed.")

# ======= LOAD MODEL SAU KHI TẢI =======
model = keras.models.load_model(MODEL_PATH)

# ======= FLASK APP =======
app = Flask(__name__)


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

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))  # Use Render's PORT env variable
#     app.run(host='0.0.0.0', port=port)
