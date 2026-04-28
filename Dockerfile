FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Cloud Run passes the port via the PORT environment variable.
# We use a default of 8080 if PORT is not set.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}