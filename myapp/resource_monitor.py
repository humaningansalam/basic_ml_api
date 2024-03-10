
from prometheus_client import Gauge
import threading
import psutil
import datetime

# # 현재 프로세스를 측정
process = psutil.Process()

# # 프로메테우스 지표 정의
app_cpu_usage = Gauge('app_cpu_usage', 'Description of CPU usage')
app_ram_usage = Gauge('app_ram_usage', 'Description of RAM usage')

def delay_5s():
    now = datetime.datetime.now()
    return 5 - now.second % 5

def monitor_app_resource():
    # CPU 사용량 측정
    cpu_usage = process.cpu_percent(interval=None)
    app_cpu_usage.set(cpu_usage)
    print(cpu_usage)

    # RAM 사용량 측정
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / (1024 ** 2) # Bytes를 MB로 변환
    app_ram_usage.set(ram_usage)


def threading_app_resource():

    thread = threading.Timer(delay_5s, threading_app_resource)
    thread.start()

    monitor_app_resource()

