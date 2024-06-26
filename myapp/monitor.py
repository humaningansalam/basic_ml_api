import os
import logging
import psutil
import threading
from prometheus_client import Gauge

class ResourceMonitor:
    def __init__(self, report_interval=5):  # 모니터링 간격을 5초로 설정
        self.app_cpu_usage = Gauge('app_cpu_usage', 'Description of CPU usage')
        self.app_ram_usage = Gauge('app_ram_usage', 'Description of RAM usage')
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

        self.report_interval = report_interval

        self.cpu_usage_samples = []
        self.ram_usage_samples = []
        self.monitoring = True

        self.sample_thread = threading.Thread(target=self.sample)
        self.sample_thread.start()


    def sample(self):
        while self.monitoring:
            cpu_usage = self.process.cpu_percent(interval=1)
            ram_usage = self.process.memory_info().rss

            logging.debug(cpu_usage)
            logging.debug(ram_usage)
            self.cpu_usage_samples.append(cpu_usage)
            self.ram_usage_samples.append(ram_usage)

    def report(self):
        logging.debug("start report")
        if self.cpu_usage_samples:  # CPU 사용량 샘플이 있을 경우에만 평균 계산
            avg_cpu_usage = sum(self.cpu_usage_samples) / len(self.cpu_usage_samples)
        else:
            avg_cpu_usage = 0  # 또는 적절한 기본값

        if self.ram_usage_samples:  # RAM 사용량 샘플이 있을 경우에만 평균 계산
            avg_ram_usage = sum(self.ram_usage_samples) / len(self.ram_usage_samples)
            avg_ram_usage = avg_ram_usage / (1024 ** 2)  # MB 단위로 변환
        else:
            avg_ram_usage = 0  # 또는 적절한 기본값

        self.app_cpu_usage.set(avg_cpu_usage)
        self.app_ram_usage.set(avg_ram_usage)

        # 샘플 리스트 초기화
        self.cpu_usage_samples = []
        self.ram_usage_samples = []

    def sample_and_report(self):
        if not self.monitoring:
            return  

        self.report()  # 리소스 사용량을 보고
        
        threading.Timer(self.report_interval, self.sample_and_report).start()


    def start_monitor(self):
        try:
            logging.debug("start_monitor")
            self.sample_and_report()  # 모니터링 시작
        except Exception as e:
            logging.error(f'error: {e}')

    def stop_monitor(self):
        self.monitoring = False