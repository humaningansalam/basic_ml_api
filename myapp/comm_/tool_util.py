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
    return datetime(now.year, now.month, now.day, 0, 0, 0) - timedelta(weeks=1)
