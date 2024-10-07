# Base image with Python 3.8 and FFmpeg installed
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Expose necessary ports
EXPOSE 5000
EXPOSE 23000

# Set the entry point to run the Flask app
CMD ["python", "app.py"]
