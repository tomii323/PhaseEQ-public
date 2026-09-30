"""Bounded metadata-only retries; no response arrays are retained here."""
import time

KEY = '_exchange_receive_failures'


def record_failure(state, assignment_id, revision, message):
    failures = state.setdefault(KEY, {})
    token = str(assignment_id)
    if failures.get(token, {}).get('revision') != revision:
        failures[token] = dict(revision=revision, attempts=0, due=time.monotonic() + 2,
                               message=str(message)[:400])
    else:
        failures[token]['message'] = str(message)[:400]


def record_success(state, assignment_id):
    state.get(KEY, {}).pop(str(assignment_id), None)


def retry_ready(root, state, channel_ids, *, now=None):
    from utils.composite_exchange import read_assignment_status
    from utils.exchange_publication import working_session_from_result, validated_fir_asset
    from composite_engine.assignment_snapshot import validated_speaker_asset
    now = time.monotonic() if now is None else now
    ready = False
    failures = state.get(KEY, {})
    for identifier, failure in list(failures.items()):
        if failure['attempts'] >= 3 or now < failure['due']:
            continue
        failure['attempts'] += 1
        failure['due'] = now + 2 ** (failure['attempts'] + 1)
        try:
            status = read_assignment_status(root, identifier)
            if status is None or status.channel_id not in channel_ids or status.revision != failure['revision']:
                failures.pop(identifier, None)
                continue
            result = status.result
            if not isinstance(result, dict):
                continue
            validated_speaker_asset(result, root, status.revision)
            if result.get('asset_schema') == 1:
                working_session_from_result(root, identifier, result)
                channel = result.get('exchange_channel')
                if isinstance(channel, dict):
                    validated_fir_asset(root, result['sample_rate_hz'], channel)
            # This only schedules application; success is recorded by its commit.
            ready = True
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return ready


def reset_retries(state):
    for failure in state.get(KEY, {}).values():
        failure.update(attempts=0, due=0)
