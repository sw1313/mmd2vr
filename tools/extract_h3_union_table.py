"""Extract the H3 Union time basis from the pruned FL2VA SafeTensors checkpoint."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


CHECKPOINT = (
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/"
    "e5eb578a89295337b8ff433a035929ce0279e0b6/"
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)


def read_bytes(checkpoint, start, size):
    if checkpoint:
        with checkpoint.open("rb") as source:
            source.seek(start)
            data = source.read(size)
    else:
        end = start + size - 1
        span = f"{start}-{end}"
        request = Request(f"{CHECKPOINT}?download=true&table_range={span}", headers={"Range": f"bytes={span}"})
        with urlopen(request, timeout=35) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {span}/"):
                raise ValueError("the server did not return the requested byte range")
            data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("incomplete checkpoint byte range")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination fl-union-table.npy")
    parser.add_argument("--checkpoint", type=Path, help="Read from an existing pruned FL2VA checkpoint")
    args = parser.parse_args()

    header_size = int.from_bytes(read_bytes(args.checkpoint, 0, 8), "little")
    if not 0 < header_size <= 16 * 1024 * 1024:
        parser.error("invalid SafeTensors header size")
    header = json.loads(read_bytes(args.checkpoint, 8, header_size))
    tensor = next((value for name, value in header.items() if name.endswith("adaln_t_table")), None)
    if tensor is None or tensor["dtype"] != "F32" or tensor["shape"] != [1025, 8]:
        parser.error("expected an F32 adaln_t_table of shape 1025 × 8")
    start, end = tensor["data_offsets"]
    if start < 0 or end - start != 1025 * 8 * 4:
        parser.error("invalid time basis byte range")
    table = np.frombuffer(read_bytes(args.checkpoint, 8 + header_size + start, end - start),
                          dtype="<f4").reshape(1025, 8)
    if not np.isfinite(table).all():
        parser.error("the time basis contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.save(output, table, allow_pickle=False)


if __name__ == "__main__":
    main()
