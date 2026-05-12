FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY AMD-OneClick/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# IMPORTANT: build with repository root as context:
#   docker build -f AMD-OneClick/Dockerfile .
COPY AMD-OneClick/app/ ./app/
COPY AMD-OneClick/templates/ ./templates/
COPY AMD-OneClick/static/ ./static/
COPY web/ ./reservation_web/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
