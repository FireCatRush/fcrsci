import inspect
import collections

def add_to_class(Class):
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

def save_hyperpara():
    frame = inspect.currentframe() # 获取当前调用堆栈
    _, _, _, local_vars = inspect.getargvalues(frame)
