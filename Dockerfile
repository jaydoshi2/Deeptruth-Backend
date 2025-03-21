# Use an official Python runtime as the base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev libpq-dev

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn (WSGI server for Django)
RUN pip install gunicorn

# Copy the entire project into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Set environment variable to ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "backend.wsgi:application"]