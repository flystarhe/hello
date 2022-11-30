import time
from datetime import datetime


def time_now(fmt=r"%Y%m%d_%H%M%S"):
    return time.strftime(fmt)


def datetime_now(fmt=r"%Y%m%d_%H%M%S_%f"):
    return datetime.now().strftime(fmt)
