# Gunakan image Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements dan install dependency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke container
COPY . .

# Expose port yang akan digunakan (ubah sesuai)
EXPOSE 8000

# Jalankan FastAPI pakai uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
