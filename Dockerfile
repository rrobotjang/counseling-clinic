FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY village/ ./village/
COPY data/ ./data/
COPY korean_chatbot_app_v2/ ./korean_chatbot_app_v2/
COPY app_clinic.py .

ENV PYTHONPATH=/app

EXPOSE 7862

CMD ["python", "app_clinic.py", "--port", "7862"]
