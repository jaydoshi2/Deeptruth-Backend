FROM python:3.10-slim

# Install system dependencies if needed
RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

# Copy dependency files and install them
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install "pymongo[srv]"  # Add this line

# Copy the application code
COPY . .

# Set environment variables (modify as needed)
ENV GUNICORN_CMD_ARGS="--timeout 120 --workers 1"

# Expose the appropriate port (if needed)
EXPOSE 8000

# Start the Gunicorn server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
