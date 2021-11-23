import os
import time
import math
from functools import wraps

import psutil


def elapsed_time(logger):
    """関数の処理時間と消費メモリを計測してlogに出力するデコレータを生成する
    Args:
        logger: loggerインスタンス
    """
    def _elapsed_time(func):
        """関数の処理時間と消費メモリを計測してlogに出力するデコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            p = psutil.Process(os.getpid())
            m0 = p.memory_info()[0] / 2. ** 30
            logger.info(f'***** Beg: {func.__name__} *****')
            v = func(*args, **kwargs)
            m1 = p.memory_info()[0] / 2. ** 30
            delta = m1 - m0
            sign = '+' if delta >= 0 else '-'
            delta = math.fabs(delta)
            logger.info(
                f'***** End: {func.__name__} {time.time() - start:.2f}sec [{m1:.1f}GB({sign}{delta:.1f}GB)] *****')
            return v
        return wrapper
    return _elapsed_time
