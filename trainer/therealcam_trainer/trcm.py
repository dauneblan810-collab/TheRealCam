"""Чтение и запись файлов моделей `.trcm` v1 (docs/MODEL_FORMAT.md §1–2).

Раскладка: ``TRCM`` | uint32 LE версия (=1) | uint32 LE длина заголовка H | JSON UTF-8 (H байт) |
нули до кратности 4 от начала файла | данные float32 LE (``dataFloats`` штук).
"""

from __future__ import annotations

import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import TrcmError

__all__ = [
    "MAGIC",
    "FORMAT_VERSION",
    "MAX_HEADER_BYTES",
    "MAX_DATA_BYTES",
    "MAX_NODES",
    "TrcmError",
    "TrcmFile",
    "encode_trcm",
    "decode_trcm",
    "read_trcm",
    "write_trcm",
    "data_offset",
]

MAGIC = b"TRCM"
FORMAT_VERSION = 1
PREFIX_BYTES = 12
MAX_HEADER_BYTES = 1 << 20  # 1 МиБ
MAX_DATA_BYTES = 64 << 20  # 64 МиБ
MAX_NODES = 256

_RESERVED_KEYS = ("tensors", "dataFloats", "dataCrc32")


@dataclass
class TrcmFile:
    """Загруженная модель: заголовок и тензоры (float32, формы из заголовка)."""

    header: dict[str, Any]
    tensors: dict[str, np.ndarray]
    data: np.ndarray  # вся секция данных, float32

    @property
    def data_floats(self) -> int:
        return int(self.data.size)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def data_offset(header_len: int) -> int:
    """Смещение секции данных: заголовок + выравнивание до 4 байт от начала файла."""
    return (PREFIX_BYTES + header_len + 3) & ~3


def encode_trcm(header: Mapping[str, Any], tensors: Mapping[str, np.ndarray]) -> bytes:
    """Собирает файл .trcm. Поля ``tensors``, ``dataFloats``, ``dataCrc32`` вычисляются здесь.

    Тензоры кладутся в секцию данных подряд, в порядке словаря ``tensors``.
    """
    hdr = {k: v for k, v in header.items() if k not in _RESERVED_KEYS}
    nodes = hdr.get("nodes", [])
    if isinstance(nodes, list) and len(nodes) > MAX_NODES:
        raise TrcmError(f"узлов {len(nodes)}, лимит формата {MAX_NODES}")

    entries = []
    chunks = []
    offset = 0
    for name, value in tensors.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0 or any(d <= 0 for d in arr.shape):
            raise TrcmError(f"тензор '{name}': недопустимая форма {list(arr.shape)}")
        if not np.all(np.isfinite(arr)):
            raise TrcmError(f"тензор '{name}' содержит NaN или бесконечность")
        entries.append({"name": str(name), "shape": [int(d) for d in arr.shape], "offset": offset})
        chunks.append(np.ascontiguousarray(arr).astype("<f4", copy=False).tobytes())
        offset += int(arr.size)
    data = b"".join(chunks)
    if len(data) > MAX_DATA_BYTES:
        raise TrcmError(f"данные {len(data)} байт больше лимита {MAX_DATA_BYTES}")

    hdr["tensors"] = entries
    hdr["dataFloats"] = offset
    hdr["dataCrc32"] = zlib.crc32(data) & 0xFFFFFFFF
    try:
        hbytes = json.dumps(hdr, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrcmError(f"заголовок нельзя записать как JSON: {exc}") from None
    if len(hbytes) > MAX_HEADER_BYTES:
        raise TrcmError(f"заголовок {len(hbytes)} байт больше лимита {MAX_HEADER_BYTES}")

    pad = data_offset(len(hbytes)) - PREFIX_BYTES - len(hbytes)
    return b"".join([MAGIC, struct.pack("<II", FORMAT_VERSION, len(hbytes)), hbytes, b"\0" * pad, data])


def decode_trcm(buf: bytes | bytearray | memoryview) -> TrcmFile:
    """Разбирает и проверяет файл .trcm (раскладка, лимиты, CRC, границы тензоров)."""
    mv = memoryview(buf)
    size = len(mv)
    if size < PREFIX_BYTES:
        raise TrcmError("файл слишком короткий для .trcm")
    if bytes(mv[0:4]) != MAGIC:
        raise TrcmError("это не файл модели .trcm (нет сигнатуры TRCM)")
    version, hlen = struct.unpack_from("<II", mv, 4)
    if version != FORMAT_VERSION:
        raise TrcmError(f"неподдерживаемая версия формата .trcm: {version} (поддерживается {FORMAT_VERSION})")
    if hlen > MAX_HEADER_BYTES:
        raise TrcmError(f"заголовок {hlen} байт больше лимита {MAX_HEADER_BYTES}")
    if size < PREFIX_BYTES + hlen:
        raise TrcmError("файл обрезан: заголовок неполный")
    try:
        header = json.loads(bytes(mv[PREFIX_BYTES : PREFIX_BYTES + hlen]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrcmError(f"заголовок не является корректным JSON в UTF-8: {exc}") from None
    if not isinstance(header, dict):
        raise TrcmError("заголовок должен быть JSON-объектом")
    if header.get("format", "trcm") != "trcm":
        raise TrcmError(f"поле format = {header.get('format')!r}, ожидается 'trcm'")
    if header.get("formatVersion", FORMAT_VERSION) != FORMAT_VERSION:
        raise TrcmError(f"formatVersion = {header.get('formatVersion')!r}, поддерживается {FORMAT_VERSION}")

    start = data_offset(hlen)
    if size < start:
        raise TrcmError("файл обрезан: нет выравнивания после заголовка")
    if any(mv[PREFIX_BYTES + hlen : start]):
        raise TrcmError("байты выравнивания после заголовка должны быть нулевыми")

    nfloats = header.get("dataFloats")
    if not _is_int(nfloats) or nfloats < 0:
        raise TrcmError("в заголовке нет корректного поля dataFloats")
    if nfloats * 4 > MAX_DATA_BYTES:
        raise TrcmError(f"данные {nfloats * 4} байт больше лимита {MAX_DATA_BYTES}")
    end = start + nfloats * 4
    if size < end:
        raise TrcmError(f"файл обрезан: ожидалось {end} байт, получено {size}")
    if size > end:
        raise TrcmError(f"лишние {size - end} байт в конце файла")

    crc = header.get("dataCrc32")
    if not _is_int(crc) or not 0 <= crc <= 0xFFFFFFFF:
        raise TrcmError("в заголовке нет корректного поля dataCrc32")
    data_bytes = mv[start:end]
    actual = zlib.crc32(data_bytes) & 0xFFFFFFFF
    if actual != crc:
        raise TrcmError(
            f"контрольная сумма данных не совпадает (dataCrc32={crc}, фактически {actual}): файл повреждён"
        )

    nodes = header.get("nodes")
    if not isinstance(nodes, list):
        raise TrcmError("в заголовке нет списка nodes")
    if len(nodes) > MAX_NODES:
        raise TrcmError(f"узлов {len(nodes)}, лимит формата {MAX_NODES}")

    data = np.frombuffer(data_bytes, dtype="<f4").astype(np.float32)
    tensor_list = header.get("tensors")
    if not isinstance(tensor_list, list):
        raise TrcmError("в заголовке нет списка tensors")
    tensors: dict[str, np.ndarray] = {}
    for i, entry in enumerate(tensor_list):
        if not isinstance(entry, dict):
            raise TrcmError(f"tensors[{i}] должен быть объектом")
        name = entry.get("name")
        shape = entry.get("shape")
        offset = entry.get("offset")
        if not isinstance(name, str) or not name:
            raise TrcmError(f"tensors[{i}]: нет имени")
        if name in tensors:
            raise TrcmError(f"тензор '{name}' объявлен дважды")
        if not isinstance(shape, list) or not shape or not all(_is_int(d) and d > 0 for d in shape):
            raise TrcmError(f"тензор '{name}': недопустимая форма {shape!r}")
        if not _is_int(offset) or offset < 0:
            raise TrcmError(f"тензор '{name}': недопустимое смещение {offset!r}")
        count = math.prod(shape)
        if offset + count > nfloats:
            raise TrcmError(
                f"тензор '{name}' выходит за пределы данных ({offset}+{count} > dataFloats={nfloats})"
            )
        tensors[name] = data[offset : offset + count].reshape(shape)
    return TrcmFile(header=header, tensors=tensors, data=data)


def read_trcm(path: str | os.PathLike) -> TrcmFile:
    """Читает и проверяет файл .trcm с диска."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise TrcmError(f"не удалось открыть файл модели {p}: {exc}") from None
    limit = data_offset(MAX_HEADER_BYTES) + MAX_DATA_BYTES
    if size > limit:
        raise TrcmError(f"файл {p} слишком большой для .trcm ({size} байт, максимум {limit})")
    return decode_trcm(p.read_bytes())


def write_trcm(path: str | os.PathLike, header: Mapping[str, Any], tensors: Mapping[str, np.ndarray]) -> bytes:
    """Записывает .trcm атомарно (через временный файл). Возвращает записанные байты."""
    blob = encode_trcm(header, tensors)
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, p)
    return blob
