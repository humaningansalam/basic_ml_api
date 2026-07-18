#common/utils
from datetime import datetime, timedelta
from pytz import timezone

def get_kr_time():
    """한국 시간 반환"""
    return datetime.now(timezone('Asia/Seoul'))

def one_week_ago():
    """일주일 전 시간 반환"""
    return get_kr_time() - timedelta(weeks=1)
