# Neural Nets — h4ckc0n (AI/ML, 200)

> There is a lot of bias in this instrument. Maybe gippity knows what to make of it. btw what are trailing zeros? ehh idc. they dont exist for me atleast tehe

**Author:** iparc
**Files:** `target.pt`, `mlp_weights.pt`
**Flag:** `d4rk{ved-in.was.here.?}`

## TL;DR

Every layer's bias, read to 5 decimal places, is a GPT-2 token ID. Decoding the biases in order spells out the flag. The weights and `target.pt` are decoys.

## 1. Looking at the files without trusting them

`.pt` files are PyTorch checkpoints: a zip archive holding a pickle (`data.pkl`) plus raw tensor storages. Unpickling an untrusted file can run arbitrary code, so I didn't call `torch.load()`. Disassembling the pickle with `pickletools` shows its structure without executing anything:

```
$ unzip -l mlp_weights.pt
  mlp_weights/data.pkl
  mlp_weights/data/0 ... mlp_weights/data/63      # 64 storages, 8 bytes each

$ python3 -m pickletools mlp_weights/data.pkl
  ...
  X  BINUNICODE 'mlp_layers.0.weight'
  c  GLOBAL     'torch DoubleStorage'
  X  BINUNICODE '0'                               # storage key
  ...
  X  BINUNICODE 'cnn_layers.0.weight'             # 1x1x1x1 conv
  ...
```

What's inside:

| File | Contents |
|---|---|
| `mlp_weights.pt` | `mlp_layers.0..24` (1×1 `Linear`) and `cnn_layers.0..6` (1×1×1×1 `Conv`). Each has one float64 weight and one float64 bias, giving 64 storages. |
| `target.pt` | one float64 tensor of shape `(1, 13)`, every value ≈ `4.8252e-06` |

Each storage is 8 bytes, a single little-endian `float64`, so `struct.unpack('<d', ...)` reads them directly. No torch needed.

## 2. The rabbit hole: the target

The obvious idea is "find the input that makes the network output `target`". With scalar layers and no activations, the whole network collapses to `y = A·x + B`. The product of 32 weights is about `1e-16`, so inverting for `x` gives garbage on the order of `1e16`. The target values are also all positive and nearly identical, which points to activations the challenge never tells you about. That route goes nowhere, so I went back to the description.

## 3. Reading the hints

* **"a lot of bias"**: look at the bias tensors, not the weights.
* **"gippity"**: GPT, so the numbers are probably GPT token IDs.
* **"trailing zeros… dont exist for me"**: the biases are stored as float32 converted to float64, so `0.00090` is actually stored as `0.0008999999845400453`. Printed naively it looks like `0.0009`, which would wrongly give token `9`. The zero is real: round to **5 decimals** and keep it.

The weights look like random init, but the biases are suspiciously clean:

```
mlp_layers.0.bias  0.0006699999794363976  -> 0.00067 -> 67
mlp_layers.1.bias  0.0001900000061141327  -> 0.00019 -> 19
mlp_layers.2.bias  0.0008099999977275729  -> 0.00081 -> 81
mlp_layers.3.bias  0.0007399999885819852  -> 0.00074 -> 74
mlp_layers.4.bias  0.0008999999845400453  -> 0.00090 -> 90
```

Every value is below 50257, the size of GPT-2's vocabulary. Low GPT-2 IDs map to single bytes (`!` is 0, so printable ASCII is `id + 33`): 67, 19, 81, 74, 90 decode to `d 4 r k {`. That's the flag format, and token 90 only works if the trailing zero is kept.

## 4. Decoding

The multi-character tokens need the real vocab, GPT-2's `encoder.json` (https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json), plus GPT-2's byte↔unicode table.

| Layer | Bias | Token ID | Text |
|---|---|---|---|
| mlp 0 | 0.00067 | 67 | `d` |
| mlp 1 | 0.00019 | 19 | `4` |
| mlp 2 | 0.00081 | 81 | `r` |
| mlp 3 | 0.00074 | 74 | `k` |
| mlp 4 | 0.00090 | 90 | `{` |
| mlp 5 | 0.01079 | 1079 | `ved` |
| mlp 6 | 0.00012 | 12 | `-` |
| mlp 7 | 0.00259 | 259 | `in` |
| mlp 8 | 0.00013 | 13 | `.` |
| mlp 9 | 0.09776 | 9776 | `was` |
| mlp 10 | 0.00013 | 13 | `.` |
| mlp 11 | 0.01456 | 1456 | `here` |
| mlp 12 | 0.40791 | 40791 | `.?` |
| mlp 13 | 0.00092 | 92 | `}` |

Everything after `}` (MLP layers 14–24 and all 7 CNN biases) decodes to filler: *" Skyrim Wars charges honest wandering conceded … Ghana proven … haha … darts sleepy"*.

```
$ python3 solve.py
...
FLAG: d4rk{ved-in.was.here.?}
```

## Solver

[`solve.py`](solve.py) needs only the standard library. It reads the checkpoint zip, uses `pickletools.genops` (no unpickling) to map each `*.bias` name to its storage key, unpacks the float64s, rounds to 5 decimals, and decodes the IDs with `encoder.json`.

```
python3 solve.py mlp_weights.pt encoder.json
```

## Takeaways

* Don't `torch.load()` challenge files unless you trust them. `pickletools` plus `struct` is enough to read a plain state dict.
* Float32 to float64 conversion adds noise like `...99998`. Round to the precision the author clearly meant, and read the hints for what that precision is.
* "Invert the network to hit the target" was a decoy. When the maths gives nonsense, go back to the hints.
