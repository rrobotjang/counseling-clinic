FROM python:3.11-slim

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY agents/ ./agents/
COPY static/ ./static/
COPY app_render_v2.py .

ENV PYTHONPATH=/app

EXPOSE 7862

CMD ["python", "app_render_v2.py", "--port", "7862"]
