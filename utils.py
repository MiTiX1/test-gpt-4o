import time
from typing import Any, TypedDict, Callable
from functools import wraps


class TimeitResponse(TypedDict):
    response: Any
    time: float


def timeit(display_time: bool = True, return_time: bool = False):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> TimeitResponse | Any:
            start = time.time()
            res = func(*args, **kwargs)
            running_time = time.time() - start
            if display_time:
                print(f"{func.__name__} took {running_time} seconds")
            if return_time:
                return {"response": res, "time": running_time}
            return res
        return wrapper
    return decorator
