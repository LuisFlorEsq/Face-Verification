# Stage 1: Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libopencv-dev \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies into a temp folder
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

WORKDIR /app

# Install only minimal system dependencies needed at runtime
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy app code
COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
