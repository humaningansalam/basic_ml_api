import os
import logging
from datetime import datetime, timedelta, time
from pytz import timezone


def get_kr_time():
    return datetime.now(timezone('Asia/Seoul'))

def delay_h(hour):
    now = get_kr_time()
    next_noon = datetime.combine(now.date() + timedelta(days=1), time(hour),tzinfo=now.tzinfo)
    return (next_noon - now).total_seconds()

def one_week_ago():
    now = get_kr_time()
    one_week_ago_time = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=now.tzinfo) - timedelta(weeks=1)
    return one_week_ago_time

def set_folder():
    try:
        os.makedirs("../data/model_", exist_ok=True)
    except OSError:
        logging.error('Error: Creating directory. ' +  "model_")

def set_logging(log_level):
    """
    setting logging
    """
    # 로그 생성
    logger = logging.getLogger()
    # 로그 레벨 문자열을 적절한 로깅 상수로 변환
    log_level_constant = getattr(logging, log_level, logging.INFO)
    # 로그의 출력 기준 설정
    logger.setLevel(log_level_constant)
    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log를 console에 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # log를 파일에 출력
    #file_handler = logging.FileHandler('GoogleTrendsBot.log')
    #file_handler.setFormatter(formatter)
    #logger.addHandler(file_handler)