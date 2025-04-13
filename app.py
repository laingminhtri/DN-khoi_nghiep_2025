import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import py7zr
from flask_cors import CORS  # Import CORS

# Tạo Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Định cấu hình CORS cho toàn bộ app

# Đường dẫn đến các phần của file nén và file mô hình sau khi giải nén
COMPRESSED_FILE_PARTS = [
    './models/best_weights_model.7z.001',
    './models/best_weights_model.7z.002',
    './models/best_weights_model.7z.003',
    './models/best_weights_model.7z.004',
]
MERGED_COMPRESSED_PATH = './models/best_weights_model.7z'
EXTRACTED_MODEL_PATH = './models/best_weights_model.keras'

# Hàm nối các phần file nén thành file .7z
def merge_file_parts():
    """
    Nối các phần file nén thành một file duy nhất.
    """
    if not os.path.exists(MERGED_COMPRESSED_PATH):
        print("Đang nối các phần file nén...")
        with open(MERGED_COMPRESSED_PATH, 'wb') as merged_file:
            for part in COMPRESSED_FILE_PARTS:
                if not os.path.exists(part):
                    raise FileNotFoundError(f"Phần file nén '{part}' không tồn tại.")
                with open(part, 'rb') as part_file:
                    merged_file.write(part_file.read())
        print("Hoàn tất nối file nén.")

# Hàm giải nén file mô hình từ file .7z
def extract_model():
    """
    Giải nén file mô hình từ file nén .7z.
    """
    if not os.path.exists(EXTRACTED_MODEL_PATH):
        print("Đang kiểm tra file nén mô hình...")
        merge_file_parts()  # Nối file trước khi giải nén
        print("Đang giải nén mô hình...")
        with py7zr.SevenZipFile(MERGED_COMPRESSED_PATH, mode='r') as archive:
            archive.extractall(path='./models')
        print("Hoàn tất giải nén mô hình.")

# Khởi tạo biến toàn cục để lưu trữ mô hình
model = None
EXTRACTED_MODEL_PATH = './models/best_weights_model.keras'


# Hàm tải mô hình
def load_model():
    """
    Tải mô hình từ file đã giải nén.
    """
    global model
    if model is None:
        print("Đang tải mô hình...")
        extract_model()
        model = tf.keras.models.load_model(EXTRACTED_MODEL_PATH)
        print("Mô hình được tải thành công.")

# Route trang chủ
@app.route('/')
def home():
    """
    Route trang chủ.
    """
    return render_template('index.html')

# Route dashboard
@app.route('/dashboard')
def dashboard():
    """
    Route Dashboard.
    """
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model()
p        rint("📦 Mô hình đã được load.")

        

        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh được gửi!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file rỗng!'}), 400

        print("Đã nhận file:", file.filename)
        print("Đang xử lý ảnh...")

        # Xử lý ảnh
        from PIL import Image
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

         print("📸 Ảnh đã được xử lý xong.")
        
        # Thực hiện dự đoán
        predictions = model.predict(img_array)
        print("Kết quả dự đoán:", predictions)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        print(f"Lỗi trong route /predict: {str(e)}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


    # Process result
        predicted_class = 'nodule' if predictions[0][0] > 0.5 else 'non-nodule'
        confidence = float(predictions[0][0]) if predictions[0][0] > 0.5 else 1 - float(predictions[0][0])

        print("✅ Phân loại:", predicted_class, "| Độ tin cậy:", confidence)

        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        print("❌ Lỗi khi dự đoán:", str(e))
        return jsonify({'error': str(e)}), 500

    # @app.route("/upload_file", methods=["POST"])
    # def upload_file():
    #     if 'file' not in request.files:
    #         return 'No file part'

    #     file = request.files['file']

    #     if file.filename == '':
    #         return 'No selected file'

    #     if file:
    #         image_path = '/content/' + file.filename
    #         file.save(image_path)  # Save the file to a folder named 'uploads'

    #     # Đọc ảnh và chuyển về kích thước mong muốn (240x240 trong trường hợp này)
    #         image = cv2.imread(image_path)
    #         image = cv2.resize(image, (240, 240))
    #         image = np.expand_dims(image, axis=0)  # Thêm chiều batch

    #     # Chuẩn hóa dữ liệu (nếu cần)
    #     # image = image / 255.0

    #     # Dự đoán nhãn
    #         prediction = best_model.predict(image)
    #         binary_prediction = np.round(prediction)

    #         return json.dumps(binary_prediction.tolist())

    #     return 'Error uploading file'


# Chạy ứng dụng Flask (chỉ dùng khi chạy cục bộ)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
