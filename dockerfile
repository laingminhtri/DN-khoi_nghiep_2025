FROM python:3.11-slim

# Cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y p7zip-full && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libc-dev \
    libcurl4-openssl-dev  # Thêm các thư viện cần thiết cho TensorFlow

# Cập nhật pip lên phiên bản mới nhất
RUN pip install --upgrade pip

# Set thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Expose cổng 5000 (hoặc cổng bạn sử dụng)
EXPOSE 5000

# Command để chạy app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
