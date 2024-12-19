FROM python:3.10-slim

ARG VERSION
ENV VERSION=$VERSION

WORKDIR /usr/src/app

COPY . .

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-root
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

# 시작 명령
CMD ["python3", "-m", "myapp.src.main"]