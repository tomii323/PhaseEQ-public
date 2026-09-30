def select_auto_crop_bands(reference_lengths, align_output_taps=False):
    """自動クロップを実行する帯域を返す。"""
    lengths = {
        str(band): max(0, int(length))
        for band, length in reference_lengths.items()
    }
    if not lengths or not align_output_taps:
        return set(lengths)

    maximum = max(lengths.values())
    return {
        band
        for band, length in lengths.items()
        if length == maximum
    }


def plan_auto_crop_candidates(
    reference_lengths,
    minimum_taps,
    align_output_taps=False,
):
    """探索対象と帯域別の初期候補タップ数を返す。"""
    search_bands = select_auto_crop_bands(
        reference_lengths,
        align_output_taps=align_output_taps,
    )
    candidates = {
        band: (
            int(minimum_taps[band])
            if band in search_bands
            else int(reference_lengths[band])
        )
        for band in reference_lengths
    }
    return search_bands, candidates
