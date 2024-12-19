FROM python:3.10-slim

ARG VERSION
ENV VERSION=$VERSION

WORKDIR /usr/src/app

COPY . .

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++ \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir poetry \
    && poetry install --no-root \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# 설치된 패키지 확인
RUN pip list

# numpy 설치 및 버전 확인
RUN python -c "import numpy; print(numpy.__version__)"

EXPOSE 5000

# 시작 명령
CMD ["python3", "-m", "myapp.src.main"]