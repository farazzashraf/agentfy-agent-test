From python:3.13.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

Expose 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]