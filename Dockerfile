FROM python:3.10.11-slim

# Avoid buffering logs
ENV PYTHONUNBUFFERED True

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
