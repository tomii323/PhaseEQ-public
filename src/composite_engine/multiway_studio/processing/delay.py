from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import math


@dataclass(frozen=True)
class OutputTiming:
    tap_count: int | None
    additional_delay_samples: float


def align_final_output_timing(channels):
    """Align nominal FIR origins after finalization without changing coefficients.

    Crop compensation already belongs to the additional DSP delay. Manual and
    measured alignment remain independent of this theoretical FIR difference.
    """
    channels = tuple(channels)
    centers = []
    arrivals = []
    for channel in channels:
        if channel.fir_stages:
            if channel.final_fir_override is None:
                raise ValueError("final coefficients are required before timing alignment")
            taps = len(channel.final_fir_override)
            if taps < 1:
                raise ValueError("final FIR must contain coefficients")
            center = (taps - 1) / 2.0
        else:
            center = 0.0
        arrival = center + float(channel.dsp_additional_delay_samples)
        if not math.isfinite(arrival):
            raise ValueError("final FIR timing must be finite")
        centers.append(center)
        arrivals.append(arrival)
    # A longer taper frame may have negative origin compensation. A common
    # causality offset keeps every exported additional Delay nonnegative.
    reference = max((*centers, *arrivals), default=0.0)
    result = []
    for channel, center, arrival in zip(channels, centers, arrivals):
        difference = reference - arrival
        metadata = dict(channel.adaptive_crop_metadata or {})
        metadata.update(
            theoretical_reference_samples=reference,
            theoretical_alignment_delay_samples=float(
                metadata.get("theoretical_alignment_delay_samples", 0.0)
            ) + difference,
        )
        result.append(replace(channel,
            dsp_additional_delay_samples=reference - center,
            adaptive_crop_metadata=metadata,
        ))
    return tuple(result)


def output_display_reference_samples(channels):
    """Remove only the shared FIR origin from display, preserving relative delays."""
    channels = tuple(channels)
    if not any(channel.fir_stages for channel in channels):
        return 0.0
    origins = []
    for channel in channels:
        center = ((len(channel.final_fir_override) - 1) / 2.0
                  if channel.fir_stages else 0.0)
        terms = (channel.adaptive_crop_metadata or {}).get('delay_breakdown', {})
        origins.append(center + float(terms.get('fir_base', channel.dsp_additional_delay_samples))
                       + float(terms.get('common', 0.0))
                       - float(channel.output_reference_delay_samples))
    if not all(math.isfinite(value) for value in origins):
        raise ValueError('display time reference must be finite')
    return min(origins, default=0.0)


def accumulate_output_delays(channels, rows, timing):
    """Commit one delay sum, separating arrival measurements from corrections."""
    rows_by_id = {str(row['channel_id']): row for row in rows}
    arrivals = timing.arrivals_by_channel if timing.complete else {}
    latest = max(arrivals.values(), default=0.0)
    plans = []
    for channel in channels:
        row = rows_by_id.get(channel.channel_id, {})
        metadata = dict(channel.adaptive_crop_metadata or {})
        previous = metadata.get('delay_breakdown', {})
        fir_base = float(previous.get('fir_base', channel.dsp_additional_delay_samples))
        automatic = float(row.get('auto_alignment_delay_samples', channel.auto_alignment_delay_samples))
        included = row.get('auto_alignment_measurement_delay_samples')
        unknown = included is None and automatic != 0.0 and channel.channel_id in arrivals
        # Untyped legacy alignment may already contain this measurement. Keep
        # it intact until re-application establishes explicit ownership.
        measured = (latest - arrivals[channel.channel_id] if channel.channel_id in arrivals and not unknown
                    else float(included or 0.0))
        residual = automatic - float(included or 0.0)
        way = float(channel.channel_relative_delay_samples)
        total = fir_base + measured + way + residual
        if not all(math.isfinite(value) for value in (total, measured, residual, fir_base, way)):
            raise ValueError('output delay components must be finite')
        plans.append((channel, metadata, dict(
            fir_base=fir_base,
            fir_origin=float(metadata.get('crop_delay_samples', 0.0)),
            fir_difference=fir_base - float(metadata.get('crop_delay_samples', 0.0)),
            speaker_arrival=arrivals.get(channel.channel_id), measurement=measured,
            alignment=residual, way=way, legacy_unclassified=unknown,
            measurement_source=(timing.source if channel.channel_id in arrivals else
                                'retained' if included else 'none'),
            total=total,
        )))
    common = max(0.0, -min((plan[2]['total'] for plan in plans), default=0.0))
    def committed_sum(terms):
        # Match the canonical/export evaluation order, including the common
        # offset inside the additional Delay field. Cancellation can otherwise
        # leave a tiny negative value rejected by DSP compatibility checks.
        return (terms['fir_base'] + terms['measurement'] + common) + terms['way'] + terms['alignment']

    negative = [terms for _, _, terms in plans if committed_sum(terms) < 0.0]
    while negative:
        common = max(
            math.nextafter(common - min(committed_sum(terms) for terms in negative), math.inf),
            *(math.nextafter(terms['fir_base'] + terms['measurement'] + common, math.inf)
              - (terms['fir_base'] + terms['measurement']) for terms in negative),
        )
        if not math.isfinite(common):
            raise ValueError('common output delay must be finite')
        negative = [terms for _, _, terms in plans if committed_sum(terms) < 0.0]
    result = []
    for channel, metadata, terms in plans:
        terms['common'] = common
        terms['total'] = committed_sum(terms)
        if not math.isfinite(terms['total']):
            raise ValueError('accumulated output delay must be finite')
        metadata['delay_breakdown'] = terms
        result.append(replace(channel,
            dsp_additional_delay_samples=terms['fir_base'] + terms['measurement'] + common,
            auto_alignment_delay_samples=terms['alignment'],
            adaptive_crop_metadata=metadata,
        ))
    return tuple(result)


def defer_output_timing(
    assigned_taps: Mapping[str, int | None],
) -> dict[str, OutputTiming]:
    """Keep design frames unchanged until final FIR lengths are known.

    Adaptive cropping changes the lengths used by the alignment policy.  Its
    FIR-length delay must therefore be resolved once, after cropping, rather
    than carrying a provisional delay into the final result.
    """
    if any(taps is not None and taps < 1 for taps in assigned_taps.values()):
        raise ValueError("assigned FIR tap counts must be positive or None")
    return {
        band: OutputTiming(taps, 0.0)
        for band, taps in assigned_taps.items()
    }


def resolve_output_timing(
    assigned_taps: Mapping[str, int | None], *, align_output_taps: bool,
) -> dict[str, OutputTiming]:
    """Plan final FIR frames and external delays from resolved assignments.

    An FIR-disabled channel cannot be padded, so it needs an external delay
    even when the FIR-enabled channels use a common frame.
    """
    if any(taps is not None and taps < 1 for taps in assigned_taps.values()):
        raise ValueError("assigned FIR tap counts must be positive or None")
    reference = max((taps or 1 for taps in assigned_taps.values()), default=1)
    result = {}
    for band, taps in assigned_taps.items():
        output_taps = reference if align_output_taps and taps is not None else taps
        padding_delay = (reference - taps) // 2 if align_output_taps and taps is not None else 0
        result[band] = OutputTiming(
            output_taps, (reference - (taps or 1)) / 2.0 - padding_delay,
        )
    return result


def fir_length_difference_delay_samples(
    tap_count: int,
    reference_tap_count: int,
    *,
    align_output_taps: bool,
) -> float:
    """Return only the external DSP delay needed to align FIR lengths.

    The intrinsic linear-phase latency ``(N - 1) / 2`` remains in the FIR
    coefficients and must not be added here a second time.
    """
    taps = int(tap_count)
    reference = int(reference_tap_count)
    if taps < 1 or reference < 1:
        raise ValueError("tap counts must be positive")
    if taps > reference:
        raise ValueError("tap count must not exceed the reference tap count")
    if align_output_taps:
        return 0.0
    return (reference - taps) / 2.0


def relativize_output_delays(channels):
    """Remove only a shared external Delay; preserve FIRs and all relative timing."""
    channels = tuple(channels)
    if not any(channel.fir_stages for channel in channels):
        return channels
    rates = {channel.sample_rate_hz for channel in channels}
    if len(rates) != 1:
        raise ValueError("relative output delays require one sample rate")
    totals = [float(channel.dsp_additional_delay_samples)
              + float(channel.channel_relative_delay_samples)
              + float(channel.auto_alignment_delay_samples) for channel in channels]
    if any(not math.isfinite(total) or total < 0.0 for total in totals):
        raise ValueError("accumulated output delays must be finite and nonnegative")
    reference = min(totals)
    result = []
    for channel, total in zip(channels, totals):
        metadata = dict(channel.adaptive_crop_metadata or {})
        metadata['output_delay_before_reference_samples'] = total
        metadata['output_reference_delay_samples'] = reference
        if isinstance(metadata.get('delay_breakdown'), dict):
            metadata['delay_breakdown'] = dict(metadata['delay_breakdown'],
                reference_reduction=reference, total_before_reference=total, total=total-reference)
        result.append(replace(channel, output_reference_delay_samples=reference,
                              adaptive_crop_metadata=metadata))
    return tuple(result)
