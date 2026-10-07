import csv
import io
import os

import numpy as np


def _read_wav(source):
    """Load the optional audio backend only for a WAV input."""
    import soundfile as sf
    try:
        return sf.read(source, dtype="float32")
    except sf.SoundFileError as exc:
        raise ValueError(f"WAVを読み込めません: {exc}") from exc


def _read_uploaded_bytes(uploaded_file):
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "getbuffer"):
        return bytes(uploaded_file.getbuffer())
    data = uploaded_file.read()
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


def _validate_fir_coefficients(data, source_name):
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{source_name}: EQ用FIRはモノラルデータを使用してください。")
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{source_name}: FIR係数がありません。")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{source_name}: NaNまたは無限大の係数は使用できません。")
    return arr


def _load_txt_fir(text, source_name):
    values = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(
                f"{source_name}: TXTの{line_no}行目を数値として読み込めません。"
            ) from exc
    return _validate_fir_coefficients(values, source_name)


def _load_csv_fir(text, source_name):
    rows = [
        row for row in csv.reader(io.StringIO(text))
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        raise ValueError(f"{source_name}: CSVにFIR係数がありません。")

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = "value" in header
    value_index = header.index("value") if has_header else (1 if len(rows[0]) >= 2 else 0)
    data_rows = rows[1:] if has_header else rows
    values = []
    for row_no, row in enumerate(data_rows, start=2 if has_header else 1):
        if value_index >= len(row):
            raise ValueError(f"{source_name}: CSVの{row_no}行目に係数列がありません。")
        try:
            values.append(float(row[value_index].strip()))
        except ValueError as exc:
            raise ValueError(
                f"{source_name}: CSVの{row_no}行目の係数を数値として読み込めません。"
            ) from exc
    return _validate_fir_coefficients(values, source_name)


def load_fir_file(uploaded_file):
    """ファイルパスまたはアップロードオブジェクトからFIR係数を読み込む。"""
    if isinstance(uploaded_file, str):
        source_name = os.path.basename(uploaded_file)
        extension = os.path.splitext(uploaded_file)[1].lower()
        if extension == ".wav":
            data, _ = _read_wav(uploaded_file)
            return _validate_fir_coefficients(data, source_name)
        if extension == ".bin":
            return _validate_fir_coefficients(
                np.fromfile(uploaded_file, dtype="<f4"), source_name
            )
        if extension in (".txt", ".csv"):
            with open(uploaded_file, "r", encoding="utf-8-sig", newline="") as file:
                text = file.read()
            return (
                _load_txt_fir(text, source_name)
                if extension == ".txt"
                else _load_csv_fir(text, source_name)
            )
    else:
        source_name = getattr(uploaded_file, "name", "uploaded_fir")
        extension = os.path.splitext(source_name)[1].lower()
        if extension == ".wav":
            data, _ = _read_wav(uploaded_file)
            return _validate_fir_coefficients(data, source_name)

        raw = _read_uploaded_bytes(uploaded_file)
        if extension == ".bin":
            if len(raw) % np.dtype("<f4").itemsize:
                raise ValueError(f"{source_name}: BINのバイト数がfloat32形式と一致しません。")
            return _validate_fir_coefficients(
                np.frombuffer(raw, dtype="<f4"), source_name
            )
        if extension in (".txt", ".csv"):
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{source_name}: UTF-8テキストとして読み込めません。") from exc
            return (
                _load_txt_fir(text, source_name)
                if extension == ".txt"
                else _load_csv_fir(text, source_name)
            )

    raise ValueError(
        f"{getattr(uploaded_file, 'name', uploaded_file)}: "
        "対応形式は .bin / .wav / .txt / .csv です。"
    )
