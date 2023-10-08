import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1_000_000  # convert to microseconds
        print(f"Function {func.__name__} took {elapsed_time:.2f} microseconds.")
        return result

    return wrapper
