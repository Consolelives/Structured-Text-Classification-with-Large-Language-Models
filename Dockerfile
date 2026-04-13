# Start from official Python 3.12 slim image — minimal
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for security — never run containers as root
RUN useradd --create-home appuser

# Copy requirements first — Docker caches this layer so reinstalls are fast
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code into the container
COPY src/ ./src/

# Give appuser ownership of all files
RUN chown -R appuser /app

# Switch to non-root user
USER appuser

# Tell Docker this container listens on port 5000
EXPOSE 5000

# The command that runs when the container starts
CMD ["python", "-m", "flask", "--app", "src/serving/app", "run", "--host=0.0.0.0", "--port=5000"]