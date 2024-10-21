import os
import logging
import psutil
import threading

from common.prometheus_metric import get_metrics

class ResourceMonitor:
    def __init__(self, report_interval=5, sample_interval=1):  # 모니터링 간격 추가
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

        self.report_interval = report_interval
        self.sample_interval = sample_interval

        self.cpu_usage_samples = []
        self.ram_usage_samples = []
        self.monitoring = True

        self.prometheus_metrics = get_metrics() 

        self.sample_thread = threading.Thread(target=self.sample)
        self.sample_thread.start()

    def sample(self):
        while self.monitoring:
            cpu_usage = self.process.cpu_percent(interval=self.sample_interval)
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

        self.prometheus_metrics.set_app_cpu_usage(avg_cpu_usage)
        self.prometheus_metrics.set_app_ram_usage(avg_ram_usage)

        # 샘플 리스트 초기화
        self.cpu_usage_samples = []
        self.ram_usage_samples = []

    def sample_and_report(self):
        if not self.monitoring:
            return  

        self.report()
        timer = threading.Timer(self.report_interval, self.sample_and_report)
        timer.setDaemon(True)  # 데몬으로 설정하여 메인 프로세스 종료 시 자동 종료
        timer.start()

    def start_monitor(self):
        try:
            logging.debug("start_monitor")
            self.sample_and_report()  # 모니터링 시작
        except Exception as e:
            logging.error(f'error: {e}')
            self.prometheus_metrics.increment_error_count('monitor_error')

    def stop_monitor(self):
        self.monitoring = False