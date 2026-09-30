from __future__ import annotations

import os

COMPOSITE_ENGINE_FLAG = "PHASEEQ_COMPOSITE_ENGINE"
NEW_COMPOSITE_ENGINE = "new_composite_engine"


def selected_engine(environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(COMPOSITE_ENGINE_FLAG, NEW_COMPOSITE_ENGINE)).strip()
    if selected != NEW_COMPOSITE_ENGINE:
        raise ValueError(
            f"{COMPOSITE_ENGINE_FLAG} must be {NEW_COMPOSITE_ENGINE}"
        )
    return selected
