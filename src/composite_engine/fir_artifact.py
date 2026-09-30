"""Confirmed final FIR shared by display, delivery and resume export."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


def source_signature(channel) -> str:
    digest = hashlib.sha256(json.dumps([
        'final-fir-a-v1', int(channel.sample_rate_hz), int(channel.tap_count or 0),
        bool(channel.remove_nyquist_enabled), float(channel.remove_nyquist_strength),
        bool(channel.cosine_taper_enabled),
    ], allow_nan=False).encode())
    if channel.pre_alignment_tap_count is not None:
        digest.update(f'pre-alignment:{channel.pre_alignment_tap_count}'.encode())
    for _, coefficients in channel.fir_stages:
        if np.iscomplexobj(coefficients):
            raise ValueError('FIR source must be real')
        values = np.asarray(coefficients, dtype='<f8')
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError('invalid FIR source')
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FinalFIRArtifact:
    source_revision: str
    coefficients: np.ndarray
    coefficient_revision: str

    @classmethod
    def capture(cls, channel, coefficients):
        values = np.asarray(coefficients, dtype='<f8')
        if values.ndim != 1 or values.size != channel.tap_count or not np.isfinite(values).all():
            raise ValueError('invalid final FIR artifact')
        payload = values.tobytes()
        return cls(source_signature(channel), np.frombuffer(payload, dtype='<f8'),
                   hashlib.sha256(payload).hexdigest())

    def validated_coefficients(self, channel):
        values = np.asarray(self.coefficients, dtype='<f8')
        if (self.source_revision != source_signature(channel)
                or values.ndim != 1 or values.size != channel.tap_count
                or not np.isfinite(values).all()
                or hashlib.sha256(values.tobytes()).hexdigest() != self.coefficient_revision):
            raise ValueError('final FIR artifact does not match current inputs')
        return values
