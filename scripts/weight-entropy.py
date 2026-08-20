#!/usr/bin/env python3
"""How much lossless compression is actually left in these weights?

Idea 2 of the batch — store weights as something smaller and reconstruct them on the
fly — is only worth building if the bytes are genuinely redundant. That is measurable
rather than arguable: parse the GGUF, find what quantisation each tensor uses, and
compute the empirical entropy of the quantised payload. Entropy is the floor for any
lossless coder, so it bounds every scheme in this family at once.

Reads only; samples blocks rather than the whole 31 GB.
"""
import collections
import math
import struct
import sys

GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, vtype):
    simple = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in simple:
        f.read(simple[vtype])
    elif vtype == 8:
        read_str(f)
    elif vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, etype)
    else:
        raise ValueError(f"unknown metadata value type {vtype}")


def entropy(counts):
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def main(path):
    f = open(path, "rb")
    magic, version = struct.unpack("<4sI", f.read(8))
    assert magic == b"GGUF", magic
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    alignment = 32
    for _ in range(n_kv):
        key = read_str(f)
        (vtype,) = struct.unpack("<I", f.read(4))
        if key.endswith("general.alignment"):
            (alignment,) = struct.unpack("<I", f.read(4))
        else:
            skip_value(f, vtype)

    tensors = []
    for _ in range(n_tensors):
        name = read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        (ttype,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        n_elem = 1
        for d in dims:
            n_elem *= d
        tensors.append((name, ttype, n_elem, offset))
    data_start = (f.tell() + alignment - 1) // alignment * alignment

    # what is this file actually made of?
    by_type = collections.Counter()
    for name, ttype, n_elem, _ in tensors:
        tname, blk, bsz = GGML_TYPES.get(ttype, (f"type{ttype}", 1, 1))
        by_type[tname] += n_elem // blk * bsz
    total = sum(by_type.values())
    print(f"{n_tensors} tensors, {total/2**30:.2f} GiB of tensor data")
    for tname, nbytes in by_type.most_common():
        print(f"  {tname:6s} {nbytes/2**30:7.2f} GiB  ({100*nbytes/total:5.1f}%)")

    # Entropy of the quantised payload, sampled. For the k-quants the payload is the
    # whole block minus its scale header; for Q8_0 it is the 32 int8 values.
    print("\nentropy of the stored bytes (bits per byte; 8.0 = incompressible):")
    for want in ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "BF16", "F16"):
        picks = [t for t in tensors if GGML_TYPES.get(t[1], ("?",))[0] == want]
        if not picks:
            continue
        counts = collections.Counter()
        sampled = 0
        for name, ttype, n_elem, offset in picks[:24]:
            _, blk, bsz = GGML_TYPES[ttype]
            nbytes = n_elem // blk * bsz
            take = min(nbytes, 4 << 20)
            f.seek(data_start + offset)
            counts.update(f.read(take))
            sampled += take
        h = entropy(counts)
        print(f"  {want:6s} {h:5.2f} bits/byte over {sampled/2**20:6.1f} MiB sampled"
              f"  -> best-case lossless saving {100*(1-h/8):4.1f}%")

    print("\nwhat that buys, against the measured wall:")
    per_gpu = 14.65
    for tname, nbytes in by_type.most_common(1):
        pass
    print(f"  weights read per GPU per token: {per_gpu:.2f} GiB at 936 GB/s = "
          f"{per_gpu*2**30/936e9*1000:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
