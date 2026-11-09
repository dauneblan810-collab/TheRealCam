import json
import struct
import zlib

import numpy as np
import pytest

from therealcam_trainer.trcm import (
    MAGIC,
    MAX_DATA_BYTES,
    MAX_HEADER_BYTES,
    MAX_NODES,
    TrcmError,
    data_offset,
    decode_trcm,
    encode_trcm,
    read_trcm,
    write_trcm,
)


def _header(name="m", nodes=None):
    return {"format": "trcm", "formatVersion": 1, "name": name, "nodes": nodes or []}


def _tensors(rng):
    return {"a.weight": rng.normal(size=(4, 3, 3, 3)).astype(np.float32),
            "a.bias": rng.normal(size=(4,)).astype(np.float32),
            "b": rng.normal(size=(2, 5)).astype(np.float32)}


def _raw(header: dict, data: bytes = b"", version: int = 1, pad_byte: bytes = b"\0") -> bytes:
    h = json.dumps(header).encode("utf-8")
    pad = data_offset(len(h)) - 12 - len(h)
    return MAGIC + struct.pack("<II", version, len(h)) + h + pad_byte * pad + data


@pytest.mark.parametrize("name", ["m", "mo", "mod", "mode", "Звёзды"])
def test_roundtrip_layout_and_alignment(name):
    rng = np.random.default_rng(0)
    tensors = _tensors(rng)
    blob = encode_trcm(_header(name), tensors)
    assert blob[:4] == b"TRCM"
    version, hlen = struct.unpack_from("<II", blob, 4)
    assert version == 1
    start = data_offset(hlen)
    assert start % 4 == 0 and blob[12 + hlen:start] == b"\0" * (start - 12 - hlen)
    header = json.loads(blob[12:12 + hlen].decode("utf-8"))
    assert header["name"] == name
    assert header["dataFloats"] == sum(t.size for t in tensors.values())
    assert header["dataCrc32"] == zlib.crc32(blob[start:])
    assert [t["offset"] for t in header["tensors"]] == [0, 108, 112]
    f = decode_trcm(blob)
    for key, value in tensors.items():
        assert f.tensors[key].dtype == np.float32
        np.testing.assert_array_equal(f.tensors[key], value)
    # данные — float32 little-endian подряд в порядке тензоров
    np.testing.assert_array_equal(np.frombuffer(blob[start:start + 4 * 108], "<f4").reshape(4, 3, 3, 3),
                                  tensors["a.weight"])


def test_write_and_read_file(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "sub" / "m.trcm"
    blob = write_trcm(path, _header(), _tensors(rng))
    assert path.read_bytes() == blob
    assert read_trcm(path).header["dataFloats"] == 122


def test_corrupted_crc_rejected():
    blob = bytearray(encode_trcm(_header(), _tensors(np.random.default_rng(2))))
    blob[-5] ^= 0x01
    with pytest.raises(TrcmError, match="контрольная сумма"):
        decode_trcm(bytes(blob))


def test_wrong_crc_in_header_rejected():
    data = np.arange(4, dtype="<f4").tobytes()
    header = _header() | {"tensors": [], "dataFloats": 4, "dataCrc32": (zlib.crc32(data) + 1) & 0xFFFFFFFF}
    with pytest.raises(TrcmError, match="контрольная сумма"):
        decode_trcm(_raw(header, data))


@pytest.mark.parametrize("mutate, message", [
    (lambda b: b"XXXX" + b[4:], "сигнатуры"),
    (lambda b: b[:4] + struct.pack("<I", 2) + b[8:], "версия"),
    (lambda b: b[:-3], "обрезан"),
    (lambda b: b + b"\0\0\0\0", "лишние"),
    (lambda b: b[:10], "короткий"),
])
def test_layout_errors(mutate, message):
    blob = encode_trcm(_header(), _tensors(np.random.default_rng(3)))
    with pytest.raises(TrcmError, match=message):
        decode_trcm(mutate(blob))


def test_nonzero_padding_rejected():
    header = _header("abc") | {"tensors": [], "dataFloats": 0, "dataCrc32": 0}
    blob = _raw(header, pad_byte=b"\x01")
    assert data_offset(len(json.dumps(header))) - 12 - len(json.dumps(header)) > 0
    with pytest.raises(TrcmError, match="выравнивания"):
        decode_trcm(blob)


def test_limits():
    huge = MAGIC + struct.pack("<II", 1, MAX_HEADER_BYTES + 1) + b"{}"
    with pytest.raises(TrcmError, match="лимит"):
        decode_trcm(huge)
    header = _header() | {"tensors": [], "dataFloats": MAX_DATA_BYTES // 4 + 1, "dataCrc32": 0}
    with pytest.raises(TrcmError, match="лимит"):
        decode_trcm(_raw(header))
    nodes = [{"op": "identity", "name": f"n{i}", "inputs": ["image"]} for i in range(MAX_NODES + 1)]
    with pytest.raises(TrcmError, match="узлов"):
        encode_trcm(_header(nodes=nodes), {})
    with pytest.raises(TrcmError, match="узлов"):
        decode_trcm(_raw(_header(nodes=nodes) | {"tensors": [], "dataFloats": 0, "dataCrc32": 0}))
    with pytest.raises(TrcmError, match="лимит"):
        encode_trcm(_header(name="x" * (MAX_HEADER_BYTES + 1)), {})


def test_tensor_validation():
    data = np.zeros(4, dtype="<f4").tobytes()
    crc = zlib.crc32(data)
    base = _header() | {"dataFloats": 4, "dataCrc32": crc}
    with pytest.raises(TrcmError, match="выходит за пределы"):
        decode_trcm(_raw(base | {"tensors": [{"name": "a", "shape": [5], "offset": 0}]}, data))
    with pytest.raises(TrcmError, match="дважды"):
        decode_trcm(_raw(base | {"tensors": [{"name": "a", "shape": [1], "offset": 0},
                                             {"name": "a", "shape": [1], "offset": 1}]}, data))
    with pytest.raises(TrcmError, match="форма"):
        decode_trcm(_raw(base | {"tensors": [{"name": "a", "shape": [0, 2], "offset": 0}]}, data))
    with pytest.raises(TrcmError, match="NaN"):
        encode_trcm(_header(), {"a": np.array([1.0, np.nan], dtype=np.float32)})


def test_bad_json_header():
    blob = MAGIC + struct.pack("<II", 1, 4) + b"\xff\xfe{}"
    with pytest.raises(TrcmError, match="JSON"):
        decode_trcm(blob)
