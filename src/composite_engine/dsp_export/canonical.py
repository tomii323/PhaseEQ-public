from __future__ import annotations

from composite_engine.fir_artifact import FinalFIRArtifact

from composite_engine.export import DSPChannelExportInput, compose_channel_fir, prepare_channel_fir

from .models import CanonicalDSPChannel, CanonicalDSPPackage


def canonical_from_export_inputs(
    channels: tuple[DSPChannelExportInput, ...], *, system_name: str,
    generated_at: str, mode: str, source_signature: str,
    workspace: dict[str, object],
) -> CanonicalDSPPackage:
    if not channels:
        raise ValueError("at least one DSP channel is required")
    sample_rate = int(channels[0].sample_rate_hz)
    result: list[CanonicalDSPChannel] = []
    for output_index, channel in enumerate(channels, start=1):
        if channel.fir_stages and int(channel.tap_count or 0) < 1:
            raise ValueError("FIR-enabled DSP channels require a positive tap_count")
        prepared = (prepare_channel_fir(channel)
                    if channel.fir_stages and channel.final_fir_override is None else None)
        final_fir = (prepared.coefficients if prepared else
                     compose_channel_fir(channel) if channel.fir_stages else None)
        crop_delay = prepared.crop_delay_samples if prepared else 0.0
        input_taps = prepared.input_taps if prepared else int(
            (channel.adaptive_crop_metadata or {}).get("original_taps",channel.tap_count or 0))
        artifact = (FinalFIRArtifact.capture(
            channel, final_fir, allow_resolved_frame=True,
            crop_delay_samples=(crop_delay if prepared else float(
                (channel.adaptive_crop_metadata or {}).get("crop_delay_samples", 0.0))),
            input_taps=input_taps,
        ) if final_fir is not None else None)
        if artifact is not None:
            final_fir = artifact.coefficients
        result.append(CanonicalDSPChannel(
            output_index=output_index, channel_id=channel.channel_id,
            name=channel.name, group=channel.group, way=channel.way,
            sample_rate_hz=sample_rate, final_fir=final_fir,
            final_fir_artifact=artifact,
            phaseeq_iir_parameters=channel.phaseeq_iir,
            phaseeq_iir_sos=tuple(tuple(float(value) for value in row) for row in getattr(channel, "phaseeq_iir_sos", ())),
            baffle_iir_parameters=(
                dict(channel.baffle_iir_parameters)
                if isinstance(channel.baffle_iir_parameters, dict) else None
            ),
            baffle_iir_sos=tuple(
                tuple(float(value) for value in row)
                for row in getattr(channel, "baffle_iir_sos", ())
            ),
            crossover=channel.iir_crossover,
            crossover_allpass=channel.auto_alignment_allpass,
            gain_db=channel.gain_db, polarity=channel.polarity,
            fir_alignment_delay_samples=channel.dsp_additional_delay_samples + crop_delay,
            manual_delay_samples=channel.channel_relative_delay_samples,
            phase_alignment_delay_samples=channel.auto_alignment_delay_samples,
            output_reference_delay_samples=channel.output_reference_delay_samples,
            timing_provenance=(
                dict(channel.timing_provenance)
                if isinstance(channel.timing_provenance, dict) else None
            ),
            adaptive_crop_metadata=(
                dict(channel.adaptive_crop_metadata)
                if isinstance(channel.adaptive_crop_metadata, dict) else None
            ),
            pre_alignment_tap_count=(int(final_fir.size) if final_fir is not None
                                     else None),
        ))
    system_payload = (
        workspace.get("multiway_system", {})
        if isinstance(workspace.get("multiway_system"), dict) else {}
    )
    return CanonicalDSPPackage(
        system_name=system_name, generated_at=generated_at, mode=mode,
        channels=tuple(result), source_signature=source_signature,
        workspace=workspace,
        system_id=str(system_payload.get("id", "")),
        management_no=str(system_payload.get("management_no", "")),
        system_revision_number=(
            int(system_payload["revision"])
            if system_payload.get("revision") is not None else None
        ),
        system_content_hash=str(system_payload.get("content_hash", "")),
    )
