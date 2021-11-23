"""Logger function.
ログ出力のための関数
"""

import sys
import logging

def get_logger(level: int = logging.INFO, out_file: str = None):
    """Get Logger Function.

    Args:
        level (int): ログの出力レベルを指定． デフォルト=logging.INFO.
        out_file (str): 出力先のファイルパスを指定． デフォルト=None．

    Returns:
        loggerを出力．

    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.hasHandlers():
        if out_file is None:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(out_file)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
