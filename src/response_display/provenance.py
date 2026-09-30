"""Display provenance only: never modifies a response or performs an FFT."""


def extension_bands(input_range, processed_range, display_range):
    if processed_range is None:
        return []
    lo, hi = map(float, display_range)
    plo, phi = map(float, processed_range)
    bands = []

    def add(start, end, label):
        start, end = max(lo, start), min(hi, end)
        if start < end:
            bands.append(dict(start=start, end=end, label=label))

    add(lo, plo, "表示専用補完")
    if input_range is not None:
        rlo, rhi = map(float, input_range)
        add(plo, rlo, "P0補完")
        add(rhi, phi, "P0補完")
    add(phi, hi, "表示専用補完")
    return bands
