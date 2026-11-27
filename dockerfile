
FROM python:3.11-slim

WORKDIR /app

# Install API runtime dependencies only
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy FastAPI application
COPY api/ ./api/

# Copy configuration
COPY config/ ./config/

# Copy trained model
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
