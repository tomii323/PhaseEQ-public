"""C-DSP delivery policy and files; reuse canonical coefficients and formatter."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace

import numpy as np

from .common import allpass_sos_rows, crossover_sos, normalized_sos, validate_package
from .models import CompatibilityIssue, CompatibilityReport, DSPProfile, ExportedDSPTarget
from .pre_dirac_biquad import pre_dirac_sos

PROFILE_ID = 'minidsp_cdsp_8x12_dl_dirac_apf'
FILENAME = 'PhaseEQ_C-DSP8x12_DiracAPF_48k.zip'
CAUTION = 'CAUTION：Pre-Dirac APFは帯域外信号を減衰させません。測定時のユニット保護はユーザー自身で行ってください。'


def cdsp_profile():
    return DSPProfile(PROFILE_ID, 'minidsp', 'miniDSP C-DSP 8x12 DL (48kHz)',
                      sample_rates_hz=(48000,), supports_fir=False,
                      options={'iir_format': 'advanced_biquad', 'xover_sections': 8,
                               'import_target': 'Outputs/Xover/Advanced Biquad', 'max_ways': 4})


def validate_cdsp(package, profile):
    # Baffle EQ is outside this delivery. Validate the exported IIR PEQ as well
    # as Xover without mutating the source package.
    xover_package = replace(package, channels=tuple(replace(channel,
        baffle_iir_parameters=None, baffle_iir_sos=()) for channel in package.channels))
    base = validate_package(xover_package, profile)
    issues = list(base.issues)
    rows = [dict(row) for row in base.channel_rows]

    def error(code, message, channel=None):
        issues.append(CompatibilityIssue('error', code, message, channel))

    settings = package.workspace.get('multiway_settings', {})
    if not settings.get('phase_alignment_auto_structure', False):
        error('automatic_phase_required', '「位相構造補償を自動適用」をONにしてください。')
    context = package.workspace.get('cdsp_export_context', {})
    if not context.get('current', False):
        error('design_pending', '設定に対応する出力を確定するため、再計算してください。')
    methods = tuple(context.get('methods', ()))
    if not methods or any(method not in {'LR4', 'NTM4'} for method in methods):
        error('unsupported_method', 'すべての境界をLR4またはNTM4に設定し、再計算してください。')
    if len(methods) > 3 or package.mode == '4Way+SUB':
        error('way_limit', '最大4帯域（3Way+SUBまたは4Way）です。帯域数を減らして再計算してください。')
    if context.get('acoustic_target', False):
        error('acoustic_target', 'IIR Auto PEQの目標をGroup Targetのみとするため、帯域分割の「音響ターゲットに反映」を全境界でOFFにし、位相・時間整合側の同名設定もOFFにして再計算してください。')
    if not package.channels:
        error('no_channels', '出力チャンネルを有効にして再計算してください。')
    try:
        dirac_targets(package)
    except (ValueError, KeyError, TypeError) as exc:
        error('dirac_target', f'Group Targetを確認してください：{exc}')
    for index, channel in enumerate(package.channels):
        cid = channel.channel_id
        peq = normalized_sos(channel.phaseeq_iir_sos)
        if len(peq) > 10:
            error('peq_capacity', f'PEQの10段上限を超えています（{len(peq)}段）。IIR設定を見直してください。', cid)
        if any(item.get('enabled', True) for item in channel.phaseeq_iir_parameters) and not len(peq):
            error('peq_pending', 'IIR PEQ係数を確定するため、PhaseEQで再生成・返却してください。', cid)
        if channel.sample_rate_hz != 48000:
            error('cdsp_rate', 'サンプルレートを48 kHzにして再計算してください。', cid)
        if channel.final_fir is not None:
            error('cdsp_fir', 'DSP FIR出力をOFFにして再計算してください。', cid)
        if not 1 <= channel.output_index <= 12:
            error('output_limit', '出力割当をCH01〜CH12以内にしてください。', cid)
        if not np.isfinite(channel.total_delay_samples) or not 0 <= channel.total_delay_samples <= 960:
            error('delay_limit', '共通のDelay基準を維持したまま、各Delayを0〜20 msに収めてください。', cid)
        if not np.isfinite(channel.gain_db) or not -72 <= channel.gain_db <= 12:
            error('gain_limit', '出力Gainを-72〜+12 dBに収めてください。', cid)
        if channel.polarity not in (-1, 1):
            error('polarity', '出力極性を+1または-1にしてください。', cid)
        try:
            pre = pre_dirac_sos(channel)
        except (ValueError, KeyError, IndexError) as exc:
            error('pre_dirac_configuration', str(exc), cid)
            continue
        post = np.vstack([crossover_sos(channel), allpass_sos_rows(channel)])
        if len(pre) > 8 or len(post) > 8:
            error('xover_capacity', f'Xoverの8段上限を超えています（Pre {len(pre)}段／Post {len(post)}段）。構成を見直してください。', cid)
        if index < len(rows):
            rows[index]['Pre Xover / 8'] = len(pre)
            rows[index]['Post Xover / 8'] = len(post)
            rows[index]['PEQ / 10'] = len(peq)
    return CompatibilityReport(profile.adapter_id, profile.profile_id, tuple(issues), tuple(rows))


def _block_biquads(sos, capacity):
    from .adapters.minidsp import _minidsp_biquads
    rows = normalized_sos(sos)
    if len(rows) > capacity:
        raise ValueError('C-DSP biquad capacity exceeded')
    identity = np.tile([1., 0., 0., 1., 0., 0.], (capacity - len(rows), 1))
    # Keep the existing sign convention and precision; the importer requires
    # no terminal comma. Fill every slot so an earlier preset cannot leak in.
    return _minidsp_biquads(np.vstack([rows, identity])).rstrip(b'\n,') + b'\n'


def cdsp_biquads(sos):
    return _block_biquads(sos, 8)


def dirac_targets(package):
    """Export only Group Target magnitude for each physical output channel."""
    from target_engine.realization import realize_target_definition
    from utils.export_bundle import target_export_text
    targets = package.workspace.get('targets', {})
    result = []
    for channel in package.channels:
        group = channel.group
        definition = targets.get(group, {}).get('preset_definition')
        if not isinstance(definition, dict):
            raise ValueError(f'{group}: target definition is missing')
        response = realize_target_definition(definition, 48000)
        path = f'Dirac_Targets/CH{channel.output_index:02d}.targetcurve'
        result.append((channel.channel_id, path, target_export_text('dirac_live_target', response).encode('utf-8')))
    return result


def export_cdsp(package, profile):
    report = validate_cdsp(package, profile)
    if not report.compatible:
        raise ValueError('; '.join(issue.message for issue in report.issues if issue.severity == 'error'))
    from .cdsp_help import cdsp_guide
    files = {'README.txt': cdsp_guide('ja').encode('utf-8-sig')}
    from phase_fir_designer.output_terms import OUTPUT_TERMS_FILENAME, OUTPUT_TERMS_BYTES
    files[OUTPUT_TERMS_FILENAME] = OUTPUT_TERMS_BYTES
    channels = []
    targets = dirac_targets(package)
    for _, path, data in targets:
        files[path] = data
    target_paths = {channel_id: path for channel_id, path, _ in targets}
    for channel in package.channels:
        stem = f'CH{channel.output_index:02d}'
        pre = pre_dirac_sos(channel)
        post = np.vstack([crossover_sos(channel), allpass_sos_rows(channel)])
        files[f'Pre_Dirac/{stem}_APF.txt'] = cdsp_biquads(pre)
        files[f'Post_Dirac/{stem}_XO.txt'] = cdsp_biquads(post)
        peq = normalized_sos(channel.phaseeq_iir_sos)
        files[f'PEQ/{stem}_PEQ.txt'] = _block_biquads(peq, 10)
        channels.append({
            'output_index': channel.output_index, 'channel_id': channel.channel_id,
            'name': channel.name, 'group': channel.group, 'way': channel.way,
            'sample_rate_hz': channel.sample_rate_hz, 'gain_db': channel.gain_db,
            'polarity': channel.polarity, 'delay_samples': channel.total_delay_samples,
            'delay_ms': channel.total_delay_samples / 48.,
            'timing_provenance': channel.timing_provenance,
            'pre_apf_sections': len(pre), 'post_xover_sections': len(post),
            'pre_file': f'Pre_Dirac/{stem}_APF.txt',
            'post_file': f'Post_Dirac/{stem}_XO.txt',
            'peq_file': f'PEQ/{stem}_PEQ.txt', 'peq_sections': len(peq),
            'dirac_target_file': target_paths[channel.channel_id],
            'measurement_phase_file': f'Pre_Dirac/{stem}_APF.txt',
        })
    files['manifest.json'] = json.dumps({
        'format': 'phaseeq-cdsp-dirac-apf', 'format_version': 3,
        'sample_rate_hz': 48000, 'fir_enabled': False,
        'automatic_phase_structure': True,
        'crossover_acoustic_target_enabled': False,
        'system_name': package.system_name, 'generated_at': package.generated_at,
        'source_signature': package.source_signature,
        'phase_reference': 'canonical_output',
        'settings_apply_to': ['Pre_Dirac', 'Post_Dirac'],
        'import_target': 'Outputs/Xover/Advanced Biquad',
        'pre_post_application': 'replace_same_slots',
        'xover_sections': 8, 'max_ways': 4,
        'peq_exported': True, 'peq_sections': 10,
        'peq_import_target': 'Outputs/PEQ',
        'peq_application': 'load_before_measurement_keep_for_playback',
        'bass_management_exported': False,
        'dirac_targets': [{'channel_id': channel.channel_id, 'output_index': channel.output_index,
                          'group': channel.group, 'file': target_paths[channel.channel_id],
                          'quantity': 'magnitude_db', 'source': 'group_target_only',
                          'normal_xover_included': False}
                          for channel in package.channels],
        'measurement_phase': 'pre_dirac_equivalent_apf_plus_automatic_phase_apf',
        'dirac_phase_workflow': 'correct_measured_response_including_pre_dirac_apf',
        'excluded_stages': ['baffle_iir'], 'channels': channels,
        'files': [{'path': path, 'sha256': hashlib.sha256(data).hexdigest()}
                  for path, data in sorted(files.items())],
    }, ensure_ascii=False, indent=2).encode()
    return ExportedDSPTarget(profile.adapter_id, profile.profile_id, files, report)


def build_cdsp_zip(exported):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path, data in sorted(exported.files.items()):
            output.writestr(path, data)
    return archive.getvalue(), FILENAME
