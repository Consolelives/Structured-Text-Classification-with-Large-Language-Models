# Use Python 3.12 slim as the base image — slim means minimal, no extras, keeps image small
FROM python:3.12-slim

# Stops Python writing .pyc compiled files inside the container — keeps it clean
ENV PYTHONDONTWRITEBYTECODE=1

# Forces Python to print logs immediately in real time — critical for debugging in production
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container — all commands run from here
WORKDIR /app

# Create a non-root user called appuser — never run containers as root for security
RUN useradd --create-home appuser

# Copy requirements first — Docker caches this layer so reinstalls are fast when code changes
COPY requirements.txt .

# Install all dependencies — no cache to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into the container — done after pip install to maximise layer caching
COPY src/ ./src/

# Give appuser ownership of everything in /app — needed before switching to non-root user
RUN chown -R appuser /app

# Switch from root to appuser — limits damage if container is ever compromised
USER appuser

# Document that the container listens on port 5000 — does not open the port, just labels it
EXPOSE 5000

# Start Gunicorn with 4 workers — production server, handles 4 simultaneous requests
# src.serving.app is the module path, :app is the Flask instance name inside that file
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "src.serving.app:app"]