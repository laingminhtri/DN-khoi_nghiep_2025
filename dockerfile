# Sử dụng Python 3.9 làm base image
FROM python:3.9-slim

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
CMD ["python", "app.py"]
