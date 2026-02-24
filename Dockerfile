# app/Dockerfile

FROM python:3.13-slim

# Set the working directory to /app
WORKDIR /app
ENV PORT=8505

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all application files from the repository
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the working directory to Pricing_app
WORKDIR /app

# Expose the Streamlit port
EXPOSE ${PORT}

# Configure health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health

# Run the Streamlit application with the new entry point
ENTRYPOINT ["streamlit", "run", "general_app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]