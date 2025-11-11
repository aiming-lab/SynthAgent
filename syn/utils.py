import os
import time
import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>', level='DEBUG')

import inspect
from functools import wraps
from contextlib import ContextDecorator
from syn.consts import const_enable_logging_stat_time, const_enable_logging_stat_time_block


def getenv_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("yes", "y", "true", "1", "t")

def stat_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if getenv_bool(const_enable_logging_stat_time):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            filename = func.__code__.co_filename
            lineno = func.__code__.co_firstlineno

            logger.debug(
                f"stat_time: Function={func.__qualname__} ({filename}:{lineno}) executed in {elapsed:.2f} seconds",
            )
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

class stat_time_block(ContextDecorator):
    def __init__(self, *, note: str = "", should_log: bool = None):
        self.note = note
        self.should_log = getenv_bool(const_enable_logging_stat_time_block) if should_log is None else should_log

        # placeholders for tracing
        self._caller_frame = None
        self._first_lineno = None
        self._max_lineno = None
        self._old_trace = None
        self._file = None
        self._start = None

    def _tracer(self, frame, event, arg):
        if event == "line" and frame is self._caller_frame:
            ln = frame.f_lineno
            if self._first_lineno is None:
                self._first_lineno = ln
            self._max_lineno = ln
        return self._tracer if self._old_trace is None else self._old_trace

    def __enter__(self):
        if not self.should_log:
            return self

        self._start = time.perf_counter()

        frame = inspect.currentframe().f_back
        self._caller_frame = frame
        self._file = frame.f_code.co_filename

        # reset line counters
        self._first_lineno = None
        self._max_lineno   = None

        self._old_trace = sys.gettrace()
        sys.settrace(self._tracer)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.should_log:
            return False

        sys.settrace(self._old_trace)

        elapsed = time.perf_counter() - self._start

        start = self._first_lineno or self._caller_frame.f_lineno

        label = f"({self._file}:{start}) " + self.note

        logger.debug(
            f"stat_time_block: Code block={label} executed in {elapsed:.4f} seconds"
        )

        return False




if __name__ == "__main__":
    pass
