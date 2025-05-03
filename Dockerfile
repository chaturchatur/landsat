# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
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

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY backend/ .

# Create necessary directories
RUN mkdir -p uploads data models sample_predictions

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0
ENV MPLBACKEND=Agg

# Create an entrypoint script
RUN echo '#!/bin/bash\nif [ "$1" = "shell" ]; then\n  /bin/bash\nelse\n  python "$1".py\nfi' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["main"]

# Expose port if needed (uncomment if your app uses a specific port)
# EXPOSE 8000 