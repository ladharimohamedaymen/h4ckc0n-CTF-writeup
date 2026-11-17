#!/usr/bin/env python3
"""DLSS solver: decrypt the museum paintings, run them through the 4xup U-Net
(re-implemented in numpy, no torch needed) and decode the resulting QR codes.

usage: python3 solve.py <museum_simulation.dist/assets> <model.pt> [outdir]
deps:  numpy, pillow, opencv-python-headless
"""
import collections, hashlib, json, os, pickle, sys, zipfile
import numpy as np
from PIL import Image
import cv2

ASSETS, MODEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "out"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. decrypt
# crypto_utils (recovered from the Nuitka constants of museum_obby.bin):
#   KEY = sha256(PASSPHRASE); blob = nonce(16) || ct
#   keystream = sha256(KEY || nonce || counter.to_bytes(4,'big')) for counter = 0,1,...
KEY = hashlib.sha256(b"unga-bunga-ved-in").digest()

def decrypt_bytes(blob: bytes) -> bytes:
    nonce, ct = blob[:16], blob[16:]
    ks = b"".join(hashlib.sha256(KEY + nonce + i.to_bytes(4, "big")).digest()
                  for i in range(len(ct) // 32 + 1))
    return bytes(a ^ b for a, b in zip(ct, ks))

# ------------------------------------------------- 2. load state_dict safely
DT = {"FloatStorage": np.float32, "LongStorage": np.int64}

def load_state_dict(path):
    z = zipfile.ZipFile(path); root = z.namelist()[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *_):
        dt, key = storage
        buf = np.frombuffer(z.read(f"{root}/data/{key}"), dtype=dt)
        if not size: return buf[offset]
        return np.lib.stride_tricks.as_strided(
            buf[offset:], shape=size, strides=[s * buf.itemsize for s in stride]).copy()
    class U(pickle.Unpickler):  # whitelist only what a plain state_dict needs
        def find_class(self, mod, name):
            if (mod, name) == ("collections", "OrderedDict"): return collections.OrderedDict
            if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"): return rebuild
            if mod == "torch" and name in DT: return name
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        def persistent_load(self, pid): return (DT[pid[1]], pid[2])
    return U(z.open(f"{root}/data.pkl")).load()["state_dict"]

SD = load_state_dict(MODEL)

# --------------------------------------------------------- 3. numpy U-Net
def conv(x, w, b=None, pad=1):
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))
    out = np.einsum("chwij,ocij->ohw", cols, w, optimize=True)
    return out + b[:, None, None] if b is not None else out

def bn(x, p):
    g, b, m, v = (SD[f"{p}.{n}"][:, None, None] for n in ("weight", "bias", "running_mean", "running_var"))
    return (x - m) / np.sqrt(v + 1e-5) * g + b

def block(x, p):  # Conv-BN-ReLU-Conv-BN-ReLU
    x = np.maximum(bn(conv(x, SD[f"{p}.block.0.weight"]), f"{p}.block.1"), 0)
    return np.maximum(bn(conv(x, SD[f"{p}.block.3.weight"]), f"{p}.block.4"), 0)

def pool(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max((2, 4))

def up(x, p):  # ConvTranspose2d(k=2, s=2)
    w, b = SD[f"{p}.weight"], SD[f"{p}.bias"]
    c, h, ww = x.shape
    return np.einsum("chw,coij->ohiwj", x, w).reshape(w.shape[1], h * 2, ww * 2) + b[:, None, None]

def unet(x):
    e1 = block(x, "enc1"); e2 = block(pool(e1), "enc2")
    e3 = block(pool(e2), "enc3"); e4 = block(pool(e3), "enc4")
    d = block(pool(e4), "bottleneck")
    for i, skip in ((4, e4), (3, e3), (2, e2), (1, e1)):
        d = block(np.concatenate([up(d, f"up{i}"), skip]), f"dec{i}")
    return conv(d, SD["head.weight"], SD["head.bias"], pad=0)[0]  # logits, 128x128

# ------------------------------------------------------------ 4. run it all
det = cv2.QRCodeDetector()
manifest = json.load(open(os.path.join(ASSETS, "manifest.json")))
for p in manifest["paintings"]:
    png = decrypt_bytes(open(os.path.join(ASSETS, p["file"]), "rb").read())
    name = p["file"].replace(".enc", "")
    open(os.path.join(OUT, name), "wb").write(png)

    img = Image.open(os.path.join(OUT, name)).convert("RGB")
    y = unet(np.asarray(img, np.float32).transpose(2, 0, 1) / 255)

    # logit > 0 means "dark module": render positive logits as black
    qr = np.where(y > 0, 0, 255).astype(np.uint8)
    qr = cv2.resize(qr, (512, 512), interpolation=cv2.INTER_NEAREST)  # blow it up 4x
    qr = cv2.copyMakeBorder(qr, 44, 44, 44, 44, cv2.BORDER_CONSTANT, value=255)  # quiet zone
    cv2.imwrite(os.path.join(OUT, name.replace(".png", "_qr.png")), qr)
    try:
        data = det.detectAndDecode(cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR))[0]
    except cv2.error:
        data = ""
    print(f"{name:14s} frame={p['frame']:7s} -> {data or '(no QR - forgery)'}")
