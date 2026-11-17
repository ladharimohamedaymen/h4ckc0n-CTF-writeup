#!/usr/bin/env python3
"""h4ckc0n - Neural Nets (AI/ML, 200) solver.

Reads the per-layer biases out of mlp_weights.pt without torch and without
unpickling (pickletools only disassembles), rounds each to 5 decimal places,
treats the digits as a GPT-2 token id and decodes them.

Usage: python3 solve.py [mlp_weights.pt] [encoder.json]
encoder.json: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
"""
import json
import pickletools
import struct
import sys
import zipfile


def load_state_dict(path):
    """Return {tensor_name: [float64 values]} for a torch zip checkpoint of DoubleStorages."""
    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]
        pkl = z.read(f"{root}/data.pkl")
        names, keys = [], []
        for op, arg, _ in pickletools.genops(pkl):
            if op.name == "BINUNICODE":
                if arg.endswith((".weight", ".bias")):
                    names.append(arg)
                elif arg.isdigit() and len(keys) < len(names):
                    keys.append(arg)  # storage key that follows the tensor name
        out = {}
        for name, key in zip(names, keys):
            raw = z.read(f"{root}/data/{key}")
            out[name] = list(struct.unpack(f"<{len(raw) // 8}d", raw))
        return out


def gpt2_decoder(encoder_path):
    enc = json.load(open(encoder_path))
    dec = {v: k for k, v in enc.items()}
    # GPT-2 byte <-> unicode table
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    return lambda ids: bytes(u2b[ch] for i in ids for ch in dec[i]).decode("utf-8", "replace")


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else "mlp_weights.pt"
    encoder = sys.argv[2] if len(sys.argv) > 2 else "encoder.json"

    sd = load_state_dict(weights)
    print(f"{len(sd)} tensors:", sorted({k.split('.')[0] for k in sd}))
    bias_names = [k for k in sd if k.endswith(".bias")]  # pickle order: mlp_layers.0..24, cnn_layers.0..6
    biases = [sd[k][0] for k in bias_names]

    # float32 -> float64 noise (0.00089999998...) disappears at 5 dp, and the
    # trailing zero is kept: 0.00090 -> 90, not 9.
    ids = [int(f"{b:.5f}".split(".")[1]) for b in biases]
    decode = gpt2_decoder(encoder)

    for name, b, t in zip(bias_names, biases, ids):
        print(f"{name:20s} bias={b:.5f}  id={t:5d}  {decode([t])!r}")

    text = decode(ids)
    flag = text[: text.index("}") + 1]
    print("\nfull decode:", repr(text))
    print("FLAG:", flag)


if __name__ == "__main__":
    main()
