"""Shared glyph-verified Japanese font discovery for Matplotlib charts."""
import os
import platform
from functools import lru_cache
from matplotlib import font_manager as fm
from matplotlib.ft2font import FT2Font

_JAPANESE_GLYPH_TEST = "日本語周波数特性"
_JAPANESE_FONT_NAME_HINTS = (
    "hiragino",
    "yu gothic",
    "yugoth",
    "meiryo",
    "ms gothic",
    "msgothic",
    "ms pgothic",
    "noto sans cjk",
    "noto sans jp",
    "source han sans",
    "ipaex",
    "ipag",
    "takao",
    "vl gothic",
    "vlgothic",
    "migu",
    "mplus",
)

_PREFERRED_JAPANESE_FONT_NAMES = {
    "Darwin": (
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "Osaka",
    ),
    "Windows": (
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "MS PGothic",
    ),
    "Linux": (
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAPGothic",
        "TakaoPGothic",
        "VL PGothic",
    ),
}

_KNOWN_JAPANESE_FONT_PATHS = {
    "Darwin": (
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
        "/System/Library/Fonts/Supplemental/YuGothic.ttc",
        "/System/Library/Fonts/Osaka.ttf",
    ),
    "Windows": (
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ),
    "Linux": (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
        "/usr/share/fonts/truetype/vlgothic/VL-PGothic-Regular.ttf",
    ),
}


def _font_supports_japanese(path):
    try:
        charmap = FT2Font(path).get_charmap()
        return all(ord(char) in charmap for char in _JAPANESE_GLYPH_TEST)
    except (OSError, RuntimeError, ValueError):
        return False


def _font_name(path):
    try:
        return fm.FontProperties(fname=path).get_name()
    except (OSError, RuntimeError, ValueError):
        return ""


def _known_japanese_font_candidates(system):
    """Return existing platform paths without initializing font discovery."""
    candidates = []
    seen = set()
    for path in (
        _KNOWN_JAPANESE_FONT_PATHS.get(system)
        or _KNOWN_JAPANESE_FONT_PATHS.get("Linux", ())
    ):
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in seen and os.path.isfile(path):
            seen.add(normalized)
            candidates.append(path)
    return candidates


def _discovered_japanese_font_candidates(system, *, excluded=()):
    """Search Matplotlib/system registries only after known paths fail."""
    preferred_names = _PREFERRED_JAPANESE_FONT_NAMES.get(
        system, _PREFERRED_JAPANESE_FONT_NAMES["Linux"]
    )
    preferred_order = {
        name.casefold(): index for index, name in enumerate(preferred_names)
    }
    paths = []
    seen = {
        os.path.normcase(os.path.abspath(path)) for path in excluded
    }

    def add(path):
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in seen and os.path.isfile(path):
            seen.add(normalized)
            paths.append(path)

    for entry in fm.fontManager.ttflist:
        name = entry.name.casefold()
        filename = os.path.basename(entry.fname).casefold()
        if (
            name in preferred_order
            or any(token in name or token.replace(" ", "") in filename
                   for token in _JAPANESE_FONT_NAME_HINTS)
        ):
            add(entry.fname)
    if not paths:
        for path in fm.findSystemFonts(fontext="ttf"):
            filename = os.path.basename(path).casefold()
            if any(token.replace(" ", "") in filename
                   for token in _JAPANESE_FONT_NAME_HINTS):
                add(path)

    def sort_key(path):
        name = _font_name(path).casefold()
        return (
            preferred_order.get(name, len(preferred_order)),
            0 if any(token in name for token in ("gothic", "sans", "meiryo")) else 1,
            name,
            path.casefold(),
        )

    return sorted(paths, key=sort_key)


def _japanese_font_candidates(system):
    """Compatibility iterator: known paths first, then discovered paths."""
    known = _known_japanese_font_candidates(system)
    return known + _discovered_japanese_font_candidates(system, excluded=known)


@lru_cache(maxsize=1)
def get_japanese_font():
    system = platform.system()
    known = _known_japanese_font_candidates(system)
    for path in known:
        if _font_supports_japanese(path):
            return fm.FontProperties(fname=path)
    for path in _discovered_japanese_font_candidates(system, excluded=known):
        if _font_supports_japanese(path):
            return fm.FontProperties(fname=path)
    return None
