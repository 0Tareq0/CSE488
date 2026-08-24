FROM python:3.10-slim

WORKDIR /app

# Copy requirements from backend
COPY backend/requirements.txt ./backend/

# Install dependencies (CPU-only PyTorch and others)
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application code
COPY backend/app/ ./backend/app/
# Copy notebook output (contains devices_embeddings.parquet)
COPY notebook/output/ ./notebook/output/

ENV PARQUET_PATH=/app/notebook/output/devices_embeddings.parquet

EXPOSE 8000

# Run uvicorn from backend directory
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
