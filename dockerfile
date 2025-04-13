FROM python:3.11-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    p7zip-full && \
    rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng 5000
EXPOSE 5000

# Chạy app với Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
