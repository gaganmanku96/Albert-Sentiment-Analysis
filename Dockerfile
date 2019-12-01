FROM python:3.6-slim

COPY api.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

RUN ["python3","api.py"]
