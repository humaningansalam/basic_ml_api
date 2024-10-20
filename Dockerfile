FROM python:3.10-slim


WORKDIR /usr/src/app

COPY . .

RUN apt-get update && apt-get install -y pkg-config libhdf5-dev gcc g++\
    && python -m pip install --upgrade pip\
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ./myapp

EXPOSE 5000

# 시작 명령
CMD ["python3", "./src/main.py"]