"""Reset only the selected topology's crossover design, retaining channel data."""
from copy import deepcopy

_BOUNDARY_FIELDS = {
    'cross_freqs', 'cycles', 'beta', 'cycles_low', 'beta_low', 'cycles_high', 'beta_high',
    'cycles_sub_low', 'beta_sub_low', 'cycles_low_mid', 'beta_low_mid',
    'cycles_mid_high', 'beta_mid_high', 'boundary_cycles', 'boundary_betas',
    'boundary_overlaps_oct', 'kaiser_overlap_oct', 'kaiser_overlap_sub_low_oct',
    'kaiser_overlap_low_mid_oct', 'kaiser_overlap_mid_high_oct', 'crossover_methods',
    'boundary_acoustic_targets', 'iir_lr2_auto_polarity', 'lr2_auto_taps',
}


def reset_mode_boundaries(settings, defaults, mode):
    result = deepcopy(settings)
    conf = result[mode]
    for key in _BOUNDARY_FIELDS:
        conf.pop(key, None)
        if key in defaults[mode]:
            conf[key] = deepcopy(defaults[mode][key])
    conf['boundary_acoustic_targets'] = [False] * len(conf['cross_freqs'])
    # The initial Kaiser topology requires FIR; retain unrelated output settings.
    if conf['cross_freqs']:
        conf['fir_output_enabled'] = True
    return result


def boundary_frequency_keys(mode, count):
    return {
        '2Way': ('fc',),
        '3Way': ('fc1', 'fc2'),
        '3Way+SUB': ('fc4_sub_low', 'fc4_low_mid', 'fc4_mid_high'),
    }.get(mode, tuple(f'boundary_fc_{mode}_{i}' for i in range(count)))
