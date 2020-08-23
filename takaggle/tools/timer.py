import time
from contextlib import contextmanager

"""使い方
with timer('message'):
    # ここに処理を記述
"""

# timerのスニペット
@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


