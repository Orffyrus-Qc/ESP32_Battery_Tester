"""
train_model.py — train the tiny feed-forward network in src/constants.h.

Forward pass (this mirrors src/constants.h EXACTLY, so the trained weights are
bit-identical to what the ESP32 computes):

    z_k = f_k * W_k + B                  (k in {0,1,2})   <-- feature-wise, not a dot product
    h_k =sigmoid(-1.25 * z_k)            (our activation is exp(-1.25 x))
    acc = sum_k( h_k * OUT_W_k )
    score = clamp(acc / 3 * 100, 0, 100)

Learned constants: ML_W[3], ML_B, ML_OUT_W[3].

Data
----
Bulk real-capacity data for primaries is not freely published, so we use a
physics-inspired capacity model over the feature cube:

    fea = f0 (= ocv / nominal_ocv),  f1 (= loaded_v / nominal_ocv, the sag term),
    f2  (= loaded_i / rated_current)

A fresh cell sits near f ~ {1.05,0.90,1.0}; an aged cell drifts to
{0.95,0.60,0.5}. We map the cube to capacity % with a smooth monotonic
function, add gentle noise, and train the net to reproduce it. Real data can
later replace make_dataset() — see the --csv option.

Usage
-----
    python tools/train_model.py                 # train + write constants.h + verify
    python tools/train_model.py --epochs 4000 --lr 0.1
"""

import argparse, re

import numpy as np


# ───────────────────────── activation + firmware forward pass ─────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.25 * np.asarray(x, dtype=np.float64)))


def firmware_score(f0, f1, f2, w, b, out):
    """Return acc and score using the element-wise firmware math.

    Matches src/constants.h: each hidden unit sees exactly ONE feature.
    """
    f = np.array([f0, f1, f2], dtype=np.float64)
    h = sigmoid(f * w + b)                      # element-wise → shape (3,)
    acc = h @ out                               # (3,).(3,) → scalar
    score = float(np.clip(acc / 3.0 * 100.0, 0.0, 100.0))
    return acc, score


# ───────────────────────── capacity (ground-truth) model ─────────────────────────
def capacity_model(f0, f1, f2):
    a = np.clip(f0, 0.0, 1.6) / 1.05
    v = np.clip(f1, 0.0, 1.6) / 0.90
    i = np.clip(f2, 0.0, 1.6) / 1.0
    mix = 0.34 * a + 0.32 * v + 0.34 * i
    return float(np.clip(100.0 * (mix ** 0.9), 0.0, 100.0))


# ───────────────────────── dataset ─────────────────────────
def make_dataset(n=6000, seed=7, spread=0.06, csv=None):
    if csv is not None:
        data = np.loadtxt(csv, delimiter=",")
        return data[:, :3], data[:, 3]
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0.90, 1.15, n)       # fresh 1.05 .. dead ~0.90
    f1 = rng.uniform(0.55, 1.00, n)       # sag (key discriminator)
    f2 = rng.uniform(0.35, 1.10, n)       # current delivery
    f0 += rng.normal(0, spread, n)
    f1 += rng.normal(0, spread, n)
    f2 += rng.normal(0, spread, n)
    f0, f1, f2 = (np.clip(x, 0.0, 1.6) for x in (f0, f1, f2))
    X = np.stack([f0, f1, f2], axis=1)
    Y = np.array([capacity_model(x0, x1, x2) for x0, x1, x2 in X], dtype=np.float64)
    return X, Y


# ───────────────────────── training (matches firmware math) ─────────────────────────
def train(X, Y, epochs=4000, lr=0.1, bs=64):
    n = X.shape[0]
    W = np.zeros(3, dtype=np.float64)
    b = 0.0
    out = np.ones(3, dtype=np.float64)     # net also learns the readout
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            bx = X[idx]                       # (bs,3)
            by = Y[idx]                       # (bs,)
            h = sigmoid(bx * W + b)           # element-wise → (bs,3)
            acc = h @ out                     # (bs,)
            score = np.clip(acc / 3.0 * 100.0, 0.0, 100.0)
            err = score - by                 # (bs,)
            # dscore/dacc = 100/3 (inside clamp); sigmoid' = 1.25*h*(1-h)
            dscore = (100.0 / 3.0) * (err / bs)       # (bs,)
            dz = dscore[:, None] * out[None, :] * (1.25 * h * (1 - h))    # (bs,3) element-wise dL/dz_k
            dW   = (dz * bx).sum(0)                       # (3,)  sum samples of dz*bx (element-wise)
            db   = dz.sum()                               # scalar (matches firmware's scalar ML_B)
            dout = (dscore[:, None] * h).sum(0)        # (bs,3).sum(0) → (3,)
            dW = np.clip(dW, -5, 5)
            db = np.clip(db, -5, 5)
            dout = np.clip(dout, -5, 5)
            W  -= lr * (dW / bs)
            b  -= lr * (db / bs)
            out -= lr * (dout / bs)
    return W, b, out


# ───────────────────────── write firmware constants ─────────────────────────
def fmt(v):
    return f"{v: 0.4f}f"


def _build_block(w, b, out):
    lines = [
        "// ───────── Micro AI model (trained by tools/train_model.py) ─────────",
        "// One tiny feed-forward net: 3 features -> hidden(3 sigmoid) -> scalar readout",
        "// → 0..100 %. Each hidden unit sees one feature (element-wise):",
        "//     z_k = F_k * W_k + B ;  h_k = sigmoid(-1.25*z_k);",
        "//     acc = sum_k( h_k * OUT_W_k );  score = clamp(acc/3*100, 0, 100).",
        "//   F0 = ocv / nominal_ocv      F1 = loaded_v / nominal_ocv",
        "//   F2 = loaded_i / rated_current  (each ~= 1.0 means nominal / fresh)",
        "// Re-train with:  python tools/train_model.py",
        "static const float ML_W[ML_HIDDEN_LAYER_SIZE]   = { " + ", ".join(fmt(float(x)) for x in w) + " };",
        "static const float ML_B                           = " + fmt(float(b)) + ";",
        "static const float ML_OUT_W[ML_HIDDEN_LAYER_SIZE] = { " + ", ".join(fmt(float(x)) for x in out) + " };",
    ]
    return "\n".join(lines)


def _decl_triple_end(src):
    """End offset of the final ML_OUT_W declaration (incl. its trailing ';').

    Robust against duplicate declaration triples and the many header/comment
    variants the file has carried over the years. We key off the last
    ML_OUT_W[...] declaration and walk back to the start of its comment block
    so the whole model region (all stale headers + every declaration line)
    vanishes in one sweep.
    """
    last_ml_out = src.rfind("ML_OUT_W[ML_HIDDEN_LAYER_SIZE]")
    if last_ml_out == -1:
        return -1
    end = src.find(";", last_ml_out)
    if end == -1:
        return -1
    return end + 1


def write_constants(path, w, b, out):
    """Rewrite the ML_* model block in `path`.

    Idempotent: removes EVERY prior AI-model region (header comment + all
    declaration lines) anywhere in the file, then writes a single clean block.
    This makes the training->write loop safe to run repeatedly even when the
    file carries duplicated or mangled blocks (the exact bug this fixes).
    """
    block = _build_block(w, b, out) + "\n"
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    hdr = src.find("Micro AI model")
    if hdr != -1:
        # Remove from the start of the header line through the last OUT_W decl.
        start = src.rfind("\n", 0, hdr) + 1
        end = _decl_triple_end(src)
        if end == -1:
            # Header, but no usable OUT_W decl; drop the whole trailing region.
            end = len(src)
        new_src = src[:start] + block + src[end:]
    else:
        # No header present now — drop every declaration triple (no header),
        # then insert one fresh block at the first declaration's position.
        decl_pat = re.compile(
            r"static const float ML_W\[ML_HIDDEN_LAYER_SIZE\] = .*?;\n"
            r"static const float ML_B[^\n]*\s*=\s*[^\n]*?;\n"
            r"static const float ML_OUT_W\[ML_HIDDEN_LAYER_SIZE\] = .*?;\n"
        )
        matches = list(decl_pat.finditer(src))
        if not matches:
            anchor = src.find("static const float ML_W")
            if anchor == -1:
                raise RuntimeError("write_constants: model anchor not found")
            new_src = src[:anchor] + block + src[anchor:]
        else:
            first_start = matches[0].start()
            tail = matches[-1].end()
            while tail < len(src) and src[tail] == "\n":
                tail += 1
            new_src = src[:first_start] + block + src[tail:]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_src)


# ───────────────────────── verify ─────────────────────────
def verify(path, w, b, out):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    end = _decl_triple_end(src)
    hdr = src.find("Micro AI model")
    start = src.rfind("\n", 0, hdr) + 1 if hdr != -1 else 0
    if end == -1:
        raise RuntimeError("verify: model block not found")
    print("\n=== model block in file ===")
    print(src[start:end])
    # sanity: score(fresh) high, score(worn) low
    _, sc_fresh = firmware_score(1.05, 0.90, 1.0, w, b, out)
    _, sc_worn  = firmware_score(0.95, 0.60, 0.5, w, b, out)
    sc_mid      = firmware_score(1.0, 0.75, 0.8, w, b, out)[1]
    print("\n=== firmware forward self-check ===")
    print(f"fresh  (1.05,0.90,1.0) -> {sc_fresh:5.2f} %")
    print(f"mid    (1.00,0.75,0.8) -> {sc_mid:5.2f} %")
    print(f"worn   (0.95,0.60,0.5) -> {sc_worn:5.2f} %")
    ok = sc_fresh > sc_mid > sc_worn and sc_worn > 5
    print(f"ordering fresh>mid>worn AND worn>5: {'OK' if ok else 'CHECK'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default="src/ESP32_Battery_Tester.ino")
    ap.add_argument("--csv", default=None, help="use real labels: col0,1,2=features, col3=capacity%")
    args = ap.parse_args()

    X, Y = make_dataset(csv=args.csv)
    print(f"Training on {X.shape[0]} samples (eps={args.epochs}, lr={args.lr})...")
    w, b, out = train(X, Y, epochs=args.epochs, lr=args.lr)
    print(f"trained ML_W  = {w}")
    print(f"trained ML_B  = {b}")
    print(f"trained OUT_W = {out}")

    # train fit over a representative subset
    rows = X[:400]
    pred = np.array([firmware_score(*x, w, b, out)[1] for x in rows])
    r2 = float(np.corrcoef(pred, Y[:400])[0, 1])
    print(f"correlation(fitted score, capacity) over 400 samples: {r2:.3f}")

    write_constants(args.out, w, b, out)
    verify(args.out, w, b, out)
    print("\nDONE — constants.h updated with trained weights.")


if __name__ == "__main__":
    main()
