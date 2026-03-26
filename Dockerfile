FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ core/
COPY middleware/ middleware/
COPY storage/ storage/

# Create data directory for ChromaDB + SQLite
RUN mkdir -p /app/data

# Run with 4 workers for concurrent prompt handling
CMD ["uvicorn", "middleware.proxy:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
