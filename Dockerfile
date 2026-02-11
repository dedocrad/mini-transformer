FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8501
EXPOSE 5000

CMD sh -c "\
    mlflow ui --host 0.0.0.0 --port 5000"
CMD sh -c "\
    mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:/mlflow/mlruns"
