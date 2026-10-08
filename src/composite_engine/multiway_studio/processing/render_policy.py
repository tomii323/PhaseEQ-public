"""Pure policy for deciding when Multiway results must be regenerated."""


def should_generate_result(
    *,
    configuration_valid: bool,
    has_cached_result: bool,
    manual_refresh: bool,
    auto_update: bool,
    signature_changed: bool,
) -> bool:
    """Return whether the current configuration requires a fresh result.

    Rendering duration is intentionally excluded.  A previous slow render may
    justify a performance notice, but must not leave controls and results out of
    sync while automatic updating is enabled.
    """
    if not configuration_valid:
        return False
    if not has_cached_result or manual_refresh:
        return True
    return bool(auto_update and signature_changed)
