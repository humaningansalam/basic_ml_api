#core/monitor
import os
import psutil
import threading
import logging
from src.common.metrics import get_metrics

class ResourceMonitor:
    def __init__(self, check_interval=5):
        """리소스 모니터링 클래스 초기화"""
        self.logger = logging.getLogger(__name__)
        self.check_interval = check_interval
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.metrics = get_metrics() # Prometheus Metrics 인스턴스
        self._stop_event = threading.Event()

    def _monitor(self):
        """주기적으로 리소스를 측정하여 메트릭에 업데이트"""
        while not self._stop_event.is_set():
            try:
                # CPU, Memory 측정
                cpu_usage = self.process.cpu_percent(interval=1)
                ram_usage = self.process.memory_info().rss / (1024 ** 2) # MB 단위 변환

                # Prometheus 업데이트
                self.metrics.set_app_cpu_usage(cpu_usage)
                self.metrics.set_app_ram_usage(ram_usage)

                # 로그 출력 (디버그용)
                self.logger.debug(f"CPU: {cpu_usage}%, RAM: {ram_usage:.2f}MB")

            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                self.metrics.increment_error_count('monitor_error')
                
            self._stop_event.wait(self.check_interval)

    def start_monitor(self):
        """모니터링 스레드 시작"""
        thread = threading.Thread(target=self._monitor, daemon=True)
        thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitor(self):
        """모니터링 중지"""
        self._stop_event.set()