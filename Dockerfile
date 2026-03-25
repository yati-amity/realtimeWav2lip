# Use a base image with Python and necessary dependencies
FROM python:3.10-slim

LABEL maintainer="krishna158@live.com"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsndfile1 \
        wget \
        gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application into the container
COPY . /app

# Expose port 8080
EXPOSE 8080

# Run the Flask application
ENTRYPOINT ["python"]
CMD ["/app/app.py"]
