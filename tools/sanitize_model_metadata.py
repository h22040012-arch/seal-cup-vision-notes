"""Remove training paths from ONNX top-level metadata without touching the graph.

Only ModelProto metadata_props (field 14) entries are edited. Unknown protobuf
fields are copied byte-for-byte. No third-party package is needed.
"""
from pathlib import Path
import argparse


def varint(data, pos):
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("invalid protobuf varint")


def encode(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def fields(data):
    pos = 0
    while pos < len(data):
        start = pos
        tag, pos = varint(data, pos)
        wire = tag & 7
        payload = None
        if wire == 0:
            _, pos = varint(data, pos)
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        elif wire == 2:
            size, pos = varint(data, pos)
            payload = data[pos:pos + size]
            pos += size
        else:
            raise ValueError("unsupported protobuf wire type")
        if pos > len(data):
            raise ValueError("truncated protobuf")
        yield tag >> 3, wire, payload, data[start:pos]


def sanitize(data):
    result = bytearray()
    for number, wire, payload, raw in fields(data):
        if number == 14 and wire == 2:
            entry = {n: p for n, w, p, _ in fields(payload) if w == 2}
            key = entry.get(1, b"").decode("utf-8")
            if key in {"description", "date"}:
                continue
        result.extend(raw)
    clean = bytes(result)
    original_graph = [p for n, w, p, _ in fields(data) if n == 7 and w == 2]
    public_graph = [p for n, w, p, _ in fields(clean) if n == 7 and w == 2]
    if len(original_graph) != 1 or original_graph != public_graph:
        raise ValueError("model graph was changed")
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("use a separate destination")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(sanitize(args.source.read_bytes()))
