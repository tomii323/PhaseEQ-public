"""Complex HP/LP requests for the FIR Gain stage, independent of UI/storage."""
import numpy as np

from crossover_engine.acoustic_target import boundary_response


GAIN_CROSSOVER_RESPONSES = {
    "linear_phase_lr2": "Linear-phase LR2",
    "linear_phase_lr4": "Linear-phase LR4",
    "iir_lr2": "IIR LR2",
    "iir_lr4": "IIR LR4",
}


def gain_crossover_response(frequency, item, sample_rate):
    frequency = np.asarray(frequency, dtype=float)
    if not item.enabled:
        return np.ones_like(frequency, dtype=complex)
    if item.response == "lr2":
        # Preserve saved legacy Gain/Target curves, which used an analog shape.
        ratio = np.maximum(frequency, 0.0) / max(float(item.fc), 1e-9)
        magnitude = 1.0 / (1.0 + ratio ** 2)
        return (1.0 - magnitude if item.mode == "hp" else magnitude).astype(complex)
    if item.response not in GAIN_CROSSOVER_RESPONSES:
        raise ValueError(f"Unsupported FIR Gain HP/LP response: {item.response}")
    return boundary_response(
        sample_rate, frequency, item.fc,
        "highpass" if item.mode == "hp" else "lowpass",
        4 if item.response.endswith("4") else 2,
        item.response.startswith("linear_phase"),
    )
