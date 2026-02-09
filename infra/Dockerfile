# Stage 1: Build Dependencies (including CUDA extensions)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel as builder

WORKDIR /app

# Install system dependencies for building C++ extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python build deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and build extensions
COPY . .
# Setting this env var forces Torch to build with CUDA support if available
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"
RUN python src/ops/setup.py build_ext --inplace

# Stage 2: Runtime Image (Slimmer)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install runtime libs (e.g. for PySpark java dependency if needed, though often separated)
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and built extensions from builder
COPY --from=builder /app/src /app/src
COPY --from=builder /app/build /app/build
# ensure src is in python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default entrypoint (can be overridden by K8s args)
CMD ["python", "src/etl/spark_job.py"]
