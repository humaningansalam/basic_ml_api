FROM python:3.10-slim


WORKDIR /usr/src/app

COPY . .

RUN python3 -m pip install  --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt 

WORKDIR ./myapp

EXPOSE 5000

# 시작 명령
CMD ["python3", "main.py"]