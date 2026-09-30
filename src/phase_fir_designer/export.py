from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
from scipy.io import wavfile


def export_wav(path: str | Path | BinaryIO, fir: np.ndarray, sample_rate: int) -> None:
    target = path if hasattr(path, "write") else Path(path)
    wavfile.write(target, sample_rate, np.asarray(fir, dtype=np.float32))


def export_txt(path: str | Path | TextIO, fir: np.ndarray) -> None:
    lines = "\n".join(f"{float(x):.10g}" for x in fir)
    if hasattr(path, "write"):
        path.write(lines + "\n")
    else:
        Path(path).write_text(lines + "\n", encoding="utf-8")


def export_csv(path: str | Path | TextIO, fir: np.ndarray) -> None:
    rows = ["index,coefficient"]
    rows.extend(f"{idx},{float(value):.10g}" for idx, value in enumerate(fir))
    content = "\n".join(rows) + "\n"
    if hasattr(path, "write"):
        path.write(content)
    else:
        Path(path).write_text(content, encoding="utf-8")


def export_bin(path: str | Path | BinaryIO, fir: np.ndarray) -> None:
    data = np.asarray(fir, dtype="<f4").tobytes()
    if hasattr(path, "write"):
        path.write(data)
    else:
        Path(path).write_bytes(data)
