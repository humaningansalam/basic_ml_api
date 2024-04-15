FROM python:3.10-slim


WORKDIR /usr/src/app

COPY . .

RUN apt-get update && apt-get install -y pkg-config libhdf5-dev gcc g++\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    &&python3 -m pip install  --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt 

WORKDIR ./myapp

EXPOSE 5000

# 시작 명령
CMD ["python3", "main.py"]