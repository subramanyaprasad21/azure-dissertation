# Dockerfile - use this exact content
FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file from repo root and install
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the container
COPY . .

# Expose port 8080 (Railway expects a port)
EXPOSE 8080

# Run the FastAPI app (app.py at repo root)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
