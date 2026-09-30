"""Band-split method policy; capability detection belongs to the caller."""
from copy import deepcopy

FIR_TO_IIR = {'linear_phase_lr2': 'iir_lr2', 'linear_phase_lr4': 'iir_lr4', 'kaiser': 'iir_lr4'}
IIR_METHODS = ('iir_lr2', 'iir_lr4', 'through')
STUDIO_METHOD_IDS = {
    'Kaiser FIR': 'kaiser', 'Linear-phase LR2 FIR': 'linear_phase_lr2',
    'Linear-phase LR4 FIR': 'linear_phase_lr4', 'LR2': 'iir_lr2',
    'LR4': 'iir_lr4', 'Through': 'through',
}


def convert_studio_methods(methods, *, fir_enabled):
    """Apply the standalone capability policy to Studio's method names."""
    names = {value: key for key, value in STUDIO_METHOD_IDS.items()}
    return [names[FIR_TO_IIR.get(STUDIO_METHOD_IDS[method], STUDIO_METHOD_IDS[method])]
            if not fir_enabled else method for method in methods]


def studio_method_options(*, fir_enabled):
    """Keep the same IIR selection order as PhaseEQ."""
    if fir_enabled:
        return list(STUDIO_METHOD_IDS)
    names = {value: key for key, value in STUDIO_METHOD_IDS.items()}
    return [names[method] for method in IIR_METHODS]


def convert_standalone_filters(items, *, fir_enabled, inherited):
    """Change only methods; never rewrite definitions owned by Multiway."""
    result = deepcopy(items)
    if not fir_enabled and not inherited:
        for item in result:
            method = item.get('response', 'kaiser')
            item['response'] = FIR_TO_IIR.get(method, method)
    return result


def acoustic_target_editable(method, *, inherited=False):
    """IIR matching may use the target even when FIR processing is disabled."""
    return not inherited and method in {
        'linear_phase_lr2', 'linear_phase_lr4', 'iir_lr2', 'iir_lr4',
    }
