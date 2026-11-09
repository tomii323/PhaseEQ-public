"""Plan supported Multiway changes, then apply them without touching channel EQ."""
from copy import deepcopy
from dataclasses import dataclass, replace

from .cdsp_dirac import PROFILE_ID


@dataclass(frozen=True)
class SetupChange:
    scope: str | None
    field: str
    before: object
    after: object
    label_ja: str
    label_en: str


def _cdsp_changes(settings, mode):
    conf = settings.get(mode, {})
    return (
        SetupChange(mode, 'fir_output_enabled', conf.get('fir_output_enabled', False), False,
                    'DSP FIR出力', 'DSP FIR output'),
        SetupChange(None, 'phase_alignment_auto_structure', settings.get('phase_alignment_auto_structure', False), True,
                    '位相構造補償を自動適用', 'Automatic structural phase compensation'),
        SetupChange(mode, 'boundary_acoustic_targets', conf.get('boundary_acoustic_targets', []),
                    [False] * len(conf.get('crossover_methods', [])),
                    '全境界の音響ターゲットに反映', 'Acoustic-target reflection at every boundary'),
        SetupChange(mode, 'phase_alignment_acoustic_target', conf.get('phase_alignment_acoustic_target', False), False,
                    '位相・時間整合の音響ターゲットに反映', 'Phase/time alignment acoustic-target reflection'),
    )


# Register additional profile-specific rules here; UI and persistence stay common.
PROFILE_SETUP_RULES = {PROFILE_ID: _cdsp_changes}


def plan_setup(settings, mode, profile):
    conf = settings.get(mode, {})
    changes = []
    if profile.sample_rates_hz and conf.get('fs') not in profile.sample_rates_hz:
        changes.append(SetupChange(mode, 'fs', conf.get('fs'), profile.sample_rates_hz[0],
                                   'サンプルレート Hz', 'Sample rate Hz'))
    rules = PROFILE_SETUP_RULES.get(profile.profile_id)
    if rules:
        changes.extend(rules(settings, mode))
    return tuple(replace(change, before=deepcopy(change.before), after=deepcopy(change.after))
                 for change in changes if change.before != change.after)


def apply_setup(settings, changes):
    updated = deepcopy(settings)
    for change in changes:
        owner = updated if change.scope is None else updated[change.scope]
        if owner.get(change.field, change.before) != change.before:
            raise ValueError('DSP setup changed; calculate a new plan before applying it')
        owner[change.field] = deepcopy(change.after)
    return updated
