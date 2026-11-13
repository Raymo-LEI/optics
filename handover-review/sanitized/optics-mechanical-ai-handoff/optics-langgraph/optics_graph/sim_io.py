"""Small, strict file boundary shared with the isolated MuJoCo worker."""
import json
import math
from pathlib import Path
import stat


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def read_bytes(path, limit=256_000):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected regular artifact file")
    with path.open("rb") as stream:
        data = stream.read() if limit is None else stream.read(limit + 1)
    if limit is not None and len(data) > limit:
        raise ValueError("Artifact exceeds size limit")
    return data


def read_json(path: Path, limit=256_000):
    if path.suffix != ".json" or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("The simulation file must be a regular .json file")
    with path.open("rb") as stream:
        raw = stream.read() if limit is None else stream.read(limit + 1)
    if limit is not None and len(raw) > limit:
        raise ValueError("Simulation JSON exceeds 256 KB")
    return decode_json(raw.decode("utf-8"))


def decode_json(text):
    def bad_constant(value):
        raise ValueError(f"Non-finite number is not supported: {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("A JSON number is outside the finite range")
        return number
    try:
        return json.loads(text, object_pairs_hook=_unique,
                          parse_constant=bad_constant, parse_float=finite_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc


def write_json(path: Path, data, *, compact=False):
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False,
                         indent=None if compact else 2).encode("utf-8")
    if len(encoded) > 256_000:
        raise ValueError("Simulation JSON exceeds 256 KB")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
    temporary.replace(path)
