"""Bounded, content-keyed caches for pure local processing (no external pickle)."""
from collections import OrderedDict
from functools import wraps
import hashlib
import logging
import pickle
from threading import RLock
from weakref import WeakValueDictionary

_LOGGER = logging.getLogger(__name__)
_MEMORY_CACHES = WeakValueDictionary()


def memory_cache_roots():
    """Diagnostic snapshots copy only keys/references, never cached payloads."""
    return {f"process.cache.{name}": function.memory_snapshot()
            for name, function in list(_MEMORY_CACHES.items())}

_SERIALIZATION_ERRORS = (pickle.PickleError, TypeError, AttributeError, ImportError, EOFError)


def content_cached(*, revision, max_entries=8, max_bytes=32 * 1024 * 1024):
    def decorate(function):
        cache = OrderedDict()
        lock = RLock()
        size = 0

        @wraps(function)
        def wrapped(*args, **kwargs):
            nonlocal size
            # Serialize every array element, including IRs excluded from dataclass
            # equality. Never deserialize user-supplied pickle files.
            try:
                source = pickle.dumps((revision, args, kwargs), protocol=5)
            except _SERIALIZATION_ERRORS as exc:
                # Hot reload can leave session objects from an older class.
                # Caching is optional; never reset the user's inputs to repair it.
                _LOGGER.debug("Input cache bypass: %s", type(exc).__name__)
                return function(*args, **kwargs)
            key = hashlib.sha256(source).digest()
            with lock:
                if key in cache:
                    payload = cache.pop(key)
                    try:
                        restored = pickle.loads(payload)
                    except _SERIALIZATION_ERRORS as exc:
                        size -= len(payload)
                        _LOGGER.debug("Cache entry discarded: %s", type(exc).__name__)
                    else:
                        cache[key] = payload
                        return restored
            result = function(*args, **kwargs)
            try:
                payload = pickle.dumps(result, protocol=5)
            except _SERIALIZATION_ERRORS as exc:
                _LOGGER.debug("Result cache bypass: %s", type(exc).__name__)
                return result
            if len(payload) <= max_bytes:
                with lock:
                    previous = cache.pop(key, None)
                    if previous is not None:
                        size -= len(previous)
                    cache[key] = payload
                    size += len(payload)
                    while len(cache) > max_entries or size > max_bytes:
                        _, removed = cache.popitem(last=False)
                        size -= len(removed)
            return result

        def clear():
            nonlocal size
            with lock:
                cache.clear()
                size = 0
        def memory_snapshot():
            with lock:
                return dict(cache)
        wrapped.memory_snapshot = memory_snapshot
        _MEMORY_CACHES[f"{function.__module__}.{function.__qualname__}"] = wrapped
        wrapped.cache_clear = clear
        return wrapped
    return decorate
