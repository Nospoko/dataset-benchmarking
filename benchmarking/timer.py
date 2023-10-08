import time
import functools
from collections import defaultdict

# Global timings dictionary
function_timings = defaultdict(list)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1_000_000  # microseconds

        # Store the timing
        function_timings[func.__name__].append(elapsed_time)

        return value

    return wrapper_timer


def reset_timings():
    function_timings.clear()
