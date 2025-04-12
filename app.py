import os
import zipfile
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

# ======= CẤU HÌNH GPU HOẶC CPU =======
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except Exception as e:
    print("Error setting GPU memory growth:", e)

# ======= GHÉP FILE MÔ HÌNH =======
MODEL_DIR = "models"
OUTPUT_ZIP = os.path.join(MODEL_DIR, "best_weights_model.zip")
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")

# Tạo thư mục models nếu chưa có
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Ghép các phần của file zip
if not os.path.exists(MODEL_PATH):
    print("Combining model parts...")
    parts = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("best_weights_model.zip.")])
    if not parts:
        raise FileNotFoundError("No parts found for the model in the 'models' directory.")

    print(f"Found parts: {parts}")
    with open(OUTPUT_ZIP, "wb") as output_file:
        for part in parts:
            part_path = os.path.join(MODEL_DIR, part)
            with open(part_path, "rb") as part_file:
                output_file.write(part_file.read())
    print("Model parts combined successfully.")

    # Kiểm tra file zip có hợp lệ không
    if not zipfile.is_zipfile(OUTPUT_ZIP):
        raise zipfile.BadZipFile("Combined file is not a valid ZIP file.")

    # Giải nén file zip để lấy file keras
    print("Extracting model...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("Model extracted successfully.")

# ======= LOAD MODEL =======
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    raise

# ======= FLASK APP =======
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    try:
        img = Image.open(file.stream).resize((224, 224))
    except Exception as e:
        return jsonify({'error': 'Invalid image file', 'message': str(e)}), 400

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    result = "nodule" if prediction[0][0] > 0.5 else "non-nodule"
    return jsonify({"result": result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
