"""Bounded, value-free diagnostics; never retain inspected objects in reports."""
from __future__ import annotations

from collections import deque
from dataclasses import fields, is_dataclass
from datetime import datetime
from functools import partial
import heapq
import io
import os
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd


def inspect_memory(roots, *, max_nodes=250_000, max_seconds=3.0, detail_limit=100):
    """Count shared identities once, including ndarray owners behind views.

    Counts are estimates of reachable Python objects, not process RSS. Unknown
    extension buffers and deliberately opaque runtime resources are excluded.
    """
    seen = set()
    end = time.monotonic() + max_seconds
    truncated = False
    details = []
    serial = 0

    def visit(value, path, depth=0):
        nonlocal truncated, serial
        if id(value) in seen:
            return 0
        if len(seen) >= max_nodes or depth > 60 or time.monotonic() >= end:
            truncated = True
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value, 0)
        logical = None
        shape = dtype = ''
        children = ()
        if isinstance(value, np.ndarray):
            logical, shape, dtype = int(value.nbytes), str(value.shape), str(value.dtype)
            # getsizeof owns the allocation only for owning arrays.
            if value.base is not None:
                children = (('base', value.base),)
            elif value.dtype.hasobject:
                children = ((str(i), item) for i, item in enumerate(value.flat))
        elif isinstance(value, memoryview):
            logical = value.nbytes
            children = (('buffer', value.obj),)
        elif isinstance(value, io.BytesIO):
            view = value.getbuffer()
            logical = view.nbytes
            view.release()
            # CPython BytesIO.__sizeof__ includes its owned buffer.
            size = max(size, logical)
        elif isinstance(value, pd.DataFrame):
            # pandas reports deep column storage; shared pandas blocks across
            # distinct frames cannot be deduplicated reliably here.
            size = int(value.memory_usage(index=True, deep=True).sum())
        elif isinstance(value, dict):
            children = ((str(key)[:100], item) for key, item in value.items())
            size += sum(sys.getsizeof(key, 0) for key in value if id(key) not in seen)
            seen.update(id(key) for key in value)
        elif isinstance(value, (list, tuple)) and value and all(type(item) in (float, int, bool) for item in value):
            # Numeric payloads can contain millions of Python scalars. Avoid
            # a diagnostic-sized identity set; count scalar slots approximately.
            size += sum(sys.getsizeof(item, 0) for item in value)
        elif isinstance(value, (list, tuple, set, frozenset, deque)):
            children = ((str(i), item) for i, item in enumerate(value))
        elif isinstance(value, partial):
            children = (('args', value.args), ('keywords', value.keywords))
        elif is_dataclass(value) and not isinstance(value, type):
            children = ((field.name, getattr(value, field.name)) for field in fields(value))
        elif not isinstance(value, (types.ModuleType, types.FunctionType, type)):
            # Only inspect already stored attributes, never properties or gc
            # referents (which can walk the entire interpreter).
            attrs = getattr(value, '__dict__', None)
            if isinstance(attrs, dict) and not type(value).__module__.startswith(('streamlit.', 'threading')):
                children = (('__dict__', attrs),)
        for name, child in children:
            size += visit(child, f'{path}.{name}', depth + 1)
            if len(seen) >= max_nodes or time.monotonic() >= end:
                truncated = True
                break
        if logical is not None or size >= 16 * 1024:
            entry = {'path': path, 'type': type(value).__name__, 'bytes': size,
                     'logical_bytes': logical, 'shape': shape, 'dtype': dtype,
                     'length': len(value) if isinstance(value, (dict, list, tuple, set, frozenset, deque)) else None}
            serial += 1
            heapq.heappush(details, (size, serial, entry))
            if len(details) > detail_limit:
                heapq.heappop(details)
        return size

    rows = []
    for name, value in sorted(roots.items()):
        before = truncated
        size = visit(value, str(name))
        rows.append({'key': str(name), 'type': type(value).__name__, 'bytes': size,
                     'partial': truncated or before})
    rows.sort(key=lambda row: row['bytes'], reverse=True)
    return {'rows': rows, 'details': [item[2] for item in sorted(details, reverse=True)],
            'total_bytes': sum(row['bytes'] for row in rows), 'objects': len(seen),
            'truncated': truncated}


def process_memory():
    result = {'pid': os.getpid(), 'timestamp': datetime.now().astimezone().isoformat()}
    try:
        text = subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'rss='],
                                       text=True, timeout=2)
        result['rss_bytes'] = int(text.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result['rss_error'] = type(exc).__name__
    try:
        import resource
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except ImportError:
        pass
    return result


def streamlit_cache_stats():
    """Serialized cache bytes only; avoid Streamlit's unbounded deep session scan."""
    try:
        from streamlit.runtime.caching.cache_data_api import get_data_cache_stats_provider
        stats = get_data_cache_stats_provider().get_stats()
        from streamlit.runtime import exists, get_instance
        if exists():
            runtime = get_instance()
            providers = [runtime.uploaded_file_mgr]
            storage = getattr(runtime.media_file_mgr, '_storage', None)
            if storage is not None and hasattr(storage, 'get_stats'):
                providers.append(storage)
            for provider in providers:
                for family, items in provider.get_stats().items():
                    stats.setdefault(family, []).extend(items)
        rows = [{'category': item.category_name, 'key': item.cache_name, 'bytes': item.byte_length}
                for items in stats.values() for item in items]
        return {'rows': sorted(rows, key=lambda row: row['bytes'], reverse=True)}
    except Exception as exc:
        return {'rows': [], 'error': type(exc).__name__}


def diagnostic_roots(state, namespace=None):
    roots = {str(key): value for key, value in state.items()
             if not str(key).startswith('_memory_diagnostic')}
    if namespace:
        roots.update({f'application.{key}': value for key, value in namespace.items()
                      if not key.startswith('__') and not key.startswith('_memory_diagnostic')
                      and isinstance(value, (dict, list, tuple, np.ndarray, pd.DataFrame, partial))})
    try:
        from response_display.cache import memory_cache_roots
        roots.update(memory_cache_roots())
    except ImportError:
        pass
    # Read already imported pyplot managers without importing a GUI backend.
    helpers = sys.modules.get('matplotlib._pylab_helpers')
    if helpers is not None:
        roots['process.matplotlib_figures'] = dict(helpers.Gcf.figs)
    return roots


def capture_memory(state, namespace=None):
    before = process_memory()
    report = inspect_memory(diagnostic_roots(state, namespace))
    report.update(process=before, process_after_scan=process_memory(), streamlit=streamlit_cache_stats())
    return report
