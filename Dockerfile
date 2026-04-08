FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY core/ core/
COPY middleware/ middleware/
COPY storage/ storage/

# Create data directory for ChromaDB
RUN mkdir -p /app/data/chromadb

# Single worker — ChromaDB PersistentClient is per-process; multiple workers
# would each open their own client. For PoC, 1 worker is simpler and correct.
CMD ["uvicorn", "middleware.proxy:app", "--host", "0.0.0.0", "--port", "8000"]
