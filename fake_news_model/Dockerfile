FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set default Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/evaluation model/checkpoints

# Set environment variables
ENV MODEL_PATH=/app/model/checkpoints/final
ENV PYTHONPATH=/app

# Expose the port the API runs on
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"] 