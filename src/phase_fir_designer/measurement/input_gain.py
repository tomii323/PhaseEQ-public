from __future__ import annotations

from dataclasses import dataclass
import ctypes
import platform
import re


@dataclass(frozen=True)
class InputGainSnapshot:
    available: bool
    device_name: str = ""
    channel_gain_db: tuple[float, ...] = ()
    channel_max_gain_db: tuple[float, ...] = ()
    channel_min_gain_db: tuple[float, ...] = ()
    analog_gain_db: float | None = None
    source: str = ""
    error: str = ""
    writable: bool = False

    @property
    def gain_adjustment_from_max_db(self) -> float | None:
        if not self.available or not self.channel_gain_db or not self.channel_max_gain_db:
            return None
        if len(self.channel_gain_db) != len(self.channel_max_gain_db):
            return None
        differences = [maximum - current for current, maximum in zip(self.channel_gain_db, self.channel_max_gain_db)]
        if max(differences) - min(differences) > 0.1:
            return None
        return float(sum(differences) / len(differences))


def read_input_gain_snapshot(device_name: str, channels: int = 2) -> InputGainSnapshot:
    if not str(device_name).strip():
        return InputGainSnapshot(available=False, error="Input device is not selected")
    if platform.system() != "Darwin":
        return InputGainSnapshot(available=False, error="Input gain reading is not implemented on this OS")
    try:
        return _read_coreaudio_input_gain(device_name, channels)
    except Exception as exc:
        return InputGainSnapshot(available=False, error=f"CoreAudio input gain could not be read: {exc}")


def set_input_gain_adjustment(device_name: str, channels: int, adjustment_db: float) -> InputGainSnapshot:
    """Adjust every exposed CoreAudio input channel by the same dB amount."""
    if platform.system() != "Darwin":
        return InputGainSnapshot(available=False, error="Automatic input gain is only implemented for CoreAudio")
    try:
        return _set_coreaudio_input_gain(device_name, channels, adjustment_db)
    except Exception as exc:
        return InputGainSnapshot(available=False, error=f"CoreAudio input gain could not be changed: {exc}")


def input_gain_is_unchanged(
    expected: InputGainSnapshot,
    current: InputGainSnapshot,
    *,
    tolerance_db: float = 0.05,
) -> bool:
    if not expected.available or not current.available:
        return False
    if _device_token(expected.device_name) != _device_token(current.device_name):
        return False
    if expected.analog_gain_db is not None or current.analog_gain_db is not None:
        if expected.analog_gain_db is None or current.analog_gain_db is None:
            return False
        if abs(expected.analog_gain_db - current.analog_gain_db) > tolerance_db:
            return False
    if len(expected.channel_gain_db) != len(current.channel_gain_db):
        return False
    return all(
        abs(before - after) <= tolerance_db
        for before, after in zip(expected.channel_gain_db, current.channel_gain_db)
    )


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


class _AudioValueRange(ctypes.Structure):
    _fields_ = [("mMinimum", ctypes.c_double), ("mMaximum", ctypes.c_double)]


def _fourcc(value: str) -> int:
    return int.from_bytes(value.encode("ascii"), "big")


def _read_coreaudio_input_gain(device_name: str, channels: int) -> InputGainSnapshot:
    coreaudio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    corefoundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    get_size = coreaudio.AudioObjectGetPropertyDataSize
    get_size.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    get_size.restype = ctypes.c_int32
    get_data = coreaudio.AudioObjectGetPropertyData
    get_data.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    get_data.restype = ctypes.c_int32
    has_property = coreaudio.AudioObjectHasProperty
    has_property.argtypes = [ctypes.c_uint32, ctypes.POINTER(_AudioObjectPropertyAddress)]
    has_property.restype = ctypes.c_ubyte
    is_settable = coreaudio.AudioObjectIsPropertySettable
    is_settable.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    is_settable.restype = ctypes.c_int32
    corefoundation.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    corefoundation.CFStringGetCString.restype = ctypes.c_ubyte
    corefoundation.CFRelease.argtypes = [ctypes.c_void_p]

    devices_address = _AudioObjectPropertyAddress(_fourcc("dev#"), _fourcc("glob"), 0)
    data_size = ctypes.c_uint32()
    if get_size(1, ctypes.byref(devices_address), 0, None, ctypes.byref(data_size)) != 0 or data_size.value == 0:
        return InputGainSnapshot(available=False, error="CoreAudio returned no devices")
    device_ids = (ctypes.c_uint32 * (data_size.value // ctypes.sizeof(ctypes.c_uint32)))()
    if get_data(1, ctypes.byref(devices_address), 0, None, ctypes.byref(data_size), device_ids) != 0:
        return InputGainSnapshot(available=False, error="CoreAudio device list could not be read")

    candidates: list[tuple[int, str]] = []
    wanted = _device_token(device_name)
    for device_id in device_ids:
        name = _coreaudio_device_name(device_id, get_data, corefoundation)
        token = _device_token(name)
        if token and (token in wanted or wanted in token or _model_token(token) == _model_token(wanted)):
            candidates.append((int(device_id), name))
    if len(candidates) != 1:
        return InputGainSnapshot(
            available=False,
            error=("CoreAudio input device was not found" if not candidates else "CoreAudio device name was ambiguous"),
        )
    device_id, resolved_name = candidates[0]
    gains: list[float] = []
    maxima: list[float] = []
    minima: list[float] = []
    writable_channels: list[bool] = []
    for element in range(1, max(1, int(channels)) + 1):
        volume_address = _AudioObjectPropertyAddress(_fourcc("vold"), _fourcc("inpt"), element)
        range_address = _AudioObjectPropertyAddress(_fourcc("vdb#"), _fourcc("inpt"), element)
        if not has_property(device_id, ctypes.byref(volume_address)):
            continue
        value = ctypes.c_float()
        value_size = ctypes.c_uint32(ctypes.sizeof(value))
        if get_data(device_id, ctypes.byref(volume_address), 0, None, ctypes.byref(value_size), ctypes.byref(value)) != 0:
            continue
        maximum = float(value.value)
        minimum = float(value.value)
        if has_property(device_id, ctypes.byref(range_address)):
            value_range = _AudioValueRange()
            range_size = ctypes.c_uint32(ctypes.sizeof(value_range))
            if get_data(
                device_id,
                ctypes.byref(range_address),
                0,
                None,
                ctypes.byref(range_size),
                ctypes.byref(value_range),
            ) == 0:
                maximum = float(value_range.mMaximum)
                minimum = float(value_range.mMinimum)
        gains.append(float(value.value))
        maxima.append(maximum)
        minima.append(minimum)
        writable = ctypes.c_ubyte()
        writable_channels.append(
            is_settable(device_id, ctypes.byref(volume_address), ctypes.byref(writable)) == 0
            and bool(writable.value)
        )
    if not gains:
        return InputGainSnapshot(available=False, device_name=resolved_name, error="Input gain property is unavailable")
    return InputGainSnapshot(
        available=True,
        device_name=resolved_name,
        channel_gain_db=tuple(gains),
        channel_max_gain_db=tuple(maxima),
        channel_min_gain_db=tuple(minima),
        analog_gain_db=_analog_gain_from_name(resolved_name),
        source="CoreAudio",
        writable=bool(writable_channels) and all(writable_channels),
    )


def _set_coreaudio_input_gain(device_name: str, channels: int, adjustment_db: float) -> InputGainSnapshot:
    before = _read_coreaudio_input_gain(device_name, channels)
    if not before.available or not before.channel_gain_db:
        return before
    coreaudio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    set_data = coreaudio.AudioObjectSetPropertyData
    set_data.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    set_data.restype = ctypes.c_int32
    is_settable = coreaudio.AudioObjectIsPropertySettable
    is_settable.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    is_settable.restype = ctypes.c_int32
    device_id = _matching_coreaudio_device_id(device_name)
    addresses = [
        _AudioObjectPropertyAddress(_fourcc("vold"), _fourcc("inpt"), index + 1)
        for index in range(len(before.channel_gain_db))
    ]
    for address in addresses:
        writable = ctypes.c_ubyte()
        if is_settable(device_id, ctypes.byref(address), ctypes.byref(writable)) != 0 or not writable.value:
            return InputGainSnapshot(
                available=False,
                device_name=before.device_name,
                error="Input gain is read-only",
            )
    lower_delta = max(
        minimum - current
        for current, minimum in zip(before.channel_gain_db, before.channel_min_gain_db)
    )
    upper_delta = min(
        maximum - current
        for current, maximum in zip(before.channel_gain_db, before.channel_max_gain_db)
    )
    applied_adjustment = _np_clip(adjustment_db, lower_delta, upper_delta)
    changed = 0
    for index, (current, address) in enumerate(zip(before.channel_gain_db, addresses)):
        minimum = before.channel_min_gain_db[index] if index < len(before.channel_min_gain_db) else current
        maximum = before.channel_max_gain_db[index] if index < len(before.channel_max_gain_db) else current
        target = float(_np_clip(current + applied_adjustment, minimum, maximum))
        value = ctypes.c_float(target)
        status = set_data(device_id, ctypes.byref(address), 0, None, ctypes.sizeof(value), ctypes.byref(value))
        if status == 0:
            changed += 1
    after = _read_coreaudio_input_gain(device_name, channels)
    if changed != len(before.channel_gain_db):
        return InputGainSnapshot(available=False, device_name=before.device_name, error="Input gain is read-only or only partly controllable")
    return after


def _matching_coreaudio_device_id(device_name: str) -> int:
    coreaudio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    corefoundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    get_size = coreaudio.AudioObjectGetPropertyDataSize
    get_data = coreaudio.AudioObjectGetPropertyData
    get_size.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    get_size.restype = ctypes.c_int32
    get_data.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    get_data.restype = ctypes.c_int32
    corefoundation.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    corefoundation.CFStringGetCString.restype = ctypes.c_ubyte
    corefoundation.CFRelease.argtypes = [ctypes.c_void_p]
    address = _AudioObjectPropertyAddress(_fourcc("dev#"), _fourcc("glob"), 0)
    size = ctypes.c_uint32()
    if get_size(1, ctypes.byref(address), 0, None, ctypes.byref(size)) != 0:
        raise RuntimeError("CoreAudio device list could not be read")
    ids = (ctypes.c_uint32 * (size.value // ctypes.sizeof(ctypes.c_uint32)))()
    if get_data(1, ctypes.byref(address), 0, None, ctypes.byref(size), ids) != 0:
        raise RuntimeError("CoreAudio device list could not be read")
    wanted = _device_token(device_name)
    matches = [int(device_id) for device_id in ids if _device_token(_coreaudio_device_name(device_id, get_data, corefoundation)) == wanted]
    if len(matches) != 1:
        raise RuntimeError("CoreAudio input device was not uniquely identified")
    return matches[0]


def _np_clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), float(minimum)), float(maximum))


def _coreaudio_device_name(device_id: int, get_data, corefoundation) -> str:
    address = _AudioObjectPropertyAddress(_fourcc("lnam"), _fourcc("glob"), 0)
    reference = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(reference))
    if get_data(device_id, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(reference)) != 0:
        return ""
    if not reference.value:
        return ""
    try:
        buffer = ctypes.create_string_buffer(1024)
        if not corefoundation.CFStringGetCString(reference, buffer, len(buffer), 0x08000100):
            return ""
        return buffer.value.decode("utf-8")
    finally:
        corefoundation.CFRelease(reference)


def _device_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _model_token(value: str) -> str:
    match = re.search(r"(?:umik|omnimic)[- ]?\d?", str(value), flags=re.IGNORECASE)
    return _device_token(match.group(0)) if match else _device_token(value)


def _analog_gain_from_name(value: str) -> float | None:
    match = re.search(r"\bGain\s*:\s*([-+]?\d+(?:\.\d+)?)\s*dB", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None
