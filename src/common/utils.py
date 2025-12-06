#common/utils
import os
import logging
from datetime import datetime, timedelta, time
from pytz import timezone

def get_kr_time():
    """한국 시간 반환"""
    return datetime.now(timezone('Asia/Seoul'))

def one_week_ago():
    """일주일 전 시간 반환"""
    now = get_kr_time()
    return datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=now.tzinfo) - timedelta(weeks=1)

def set_folder(path):
    """폴더 생성"""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        logging.error(f'Error: Creating directory. {path}')

def delay_h(hour):
    """지정된 시간까지 대기"""
    now = get_kr_time()
    next_noon = datetime.combine(now.date() + timedelta(days=1), time(hour), tzinfo=now.tzinfo)
    return (next_noon - now).total_seconds()