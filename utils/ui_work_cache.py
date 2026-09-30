"""Bounded reusable preparation; persistence and UI delivery stay outside."""
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import hashlib
import json
import pickle

from response_display.cache import content_cached
from utils.settings_io import build_config_payload, processing_manifest, payload_revision


def deferred_call(function, *args, **kwargs):
    """Freeze confirmed inputs on the UI thread; run pure work only on download."""
    args, kwargs = deepcopy((args, kwargs))
    return partial(function, *args, **kwargs)


def materialize_config(arguments, extra_io):
    builder = build_config_payload if arguments['embed_response_data'] else cached_config_payload
    payload = builder(**arguments)
    payload.setdefault('io', {}).update(extra_io)
    if 'speaker_impulse_raw' in extra_io:
        payload['revision'] = payload_revision(payload)
    return payload


def download_settings(factory):
    return json.dumps(factory(), ensure_ascii=False, indent=2)


def working_session(factory, arguments):
    from utils.export_bundle import build_project_zip
    return build_project_zip(settings_payload=factory(), **arguments)


def archive_data(factory):
    return factory()[1]


def first_result(function, *args):
    """Adapt exports returning (bytes, filename) to deferred download data."""
    return function(*args)[0]


def download_fir(format_name, fir, sample_rate, options):
    from phase_fir_designer.export_pipeline import output_fir_for_export_report
    from utils.export_bundle import fir_export_bytes
    report = output_fir_for_export_report(fir, **options)
    return fir_export_bytes(format_name, report.fir, sample_rate)


cached_processing_manifest = content_cached(revision="save-manifest-v1")(processing_manifest)
cached_config_payload = content_cached(revision="save-payload-v1")(build_config_payload)


@content_cached(revision="save-json-v1", max_entries=2)
def settings_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def prepared_value(state, namespace, inputs, compute, *, max_entries=8, max_bytes=16 * 1024 * 1024,
                   independent_parts=False):
    """Session-local content cache. Never use identities or untrusted pickle."""
    errors = (pickle.PickleError, TypeError, AttributeError, ImportError, EOFError)
    try:
        # Value-only chart arguments must not depend on pickle memo sharing
        # between otherwise equal, independently produced display inputs.
        source = (b''.join(hashlib.sha256(pickle.dumps(part, protocol=5)).digest()
                           for part in inputs) if independent_parts
                  else pickle.dumps(inputs, protocol=5))
        key = hashlib.sha256(source).digest()
    except errors:
        return compute()
    entries = state.setdefault(namespace, OrderedDict())
    payload = entries.pop(key, None)
    if payload is not None:
        try:
            value = pickle.loads(payload)
        except errors:
            pass
        else:
            entries[key] = payload
            return value
    value = compute()
    if value is None:
        return value  # Don't hide warning-only render paths on the next run.
    try:
        payload = pickle.dumps(value, protocol=5)
    except errors:
        return value
    if len(payload) <= max_bytes:
        entries[key] = payload
        while len(entries) > max_entries or sum(map(len, entries.values())) > max_bytes:
            entries.popitem(last=False)
    return value
