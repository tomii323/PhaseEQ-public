"""Load and combine optional EQ FIRs without changing saved file references."""
from pathlib import Path

from fir_design_common import compose_fir_stages_full
from ..utils.io import load_fir_file


def load_eq_firs(bands, files, directory, *, enabled):
    result = dict.fromkeys(bands)
    if not enabled:
        return result
    for band in bands:
        for index, name in enumerate(files.get(band, ())):
            if not name:
                continue
            path = Path(directory) / f'{band}_eq{index+1}_{name}'
            if path.exists():
                coefficients = load_fir_file(str(path)).flatten()
                previous = result[band]
                result[band] = coefficients if previous is None else compose_fir_stages_full(
                    (previous, coefficients),
                )
    return result
