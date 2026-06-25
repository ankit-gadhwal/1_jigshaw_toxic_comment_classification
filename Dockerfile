FROM python:3.13-slim

WORKDIR /app

COPY docker_requirements.txt .

RUN pip install --no-cache-dir -r docker_requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.address=0.0.0.0"]
