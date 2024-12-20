# Use Python 3.10 as the base image
FROM python:3.10

# Install required system dependencies for TensorFlow and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgomp1 \
    curl \
    && apt-get clean

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt ./

# Install TensorFlow and its dependencies first
RUN pip install --no-cache-dir six
RUN pip install --no-cache-dir tensorflow==2.18.0 --no-deps
RUN pip install --no-cache-dir "numpy>=1.26.0,<2.1.0"
RUN pip install --no-cache-dir "scipy>=1.10.1,<1.11.0"
RUN pip install --no-cache-dir keras==2.8.0

# Install remaining dependencies (ignoring TensorFlow-related numpy conflicts)
RUN pip cache purge
RUN pip install --no-cache-dir -r requirements.txt --no-deps

# Copy application code
COPY main.py ./
COPY event_detector.py ./
COPY predict_ner.py ./
COPY data/ ./data/

# Expose necessary ports
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py", "ws://143.110.238.245:8000/stream"]
