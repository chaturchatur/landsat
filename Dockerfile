# Use Python 3.9 as base image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code, data, and models into the image
COPY backend/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0
ENV MPLBACKEND=Agg

# Default command: run download_data.py, then main.py, then result.py
CMD ["/bin/bash", "-c", "python download_data.py && python main.py && python result.py"]