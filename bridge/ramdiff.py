"""Find RAM addresses by differencing dumps, and extract physics from a series.

Two modes:
  pair   : two dumps + the expected direction of motion -> position candidates.
  series : N dumps in chronological order -> the address whose sequence is a
           perfect parabola (constant acceleration). That address is the vertical
           position/velocity and the fitted acceleration is the game's gravity.

Nothing here depends on knowing the game; it is statistics over bytes.

Scope limit found the hard way: `candidates_series` fits ONE quadratic to the
whole series, so it is only valid for a single excursion. Fed a series with
repeated jumps it correctly fails, which looks like a broken tool but is the
model being wrong. For repeated arcs use `arcs.py` instead.
"""
from __future__ import annotations
import copy
import json
import struct
import sys

BASE = 0x02000000
# 0x1000 == 1 pixel. The earlier value here (65536, i.e. 16.16) was wrong by 16x
# and every pixel figure produced by this module inherited that error. Evidence
# for 12 fractional bits: src/AAA.hpp:17 `_FixedFlt(flt) ((i32)(flt * 4096.0))`,
# src/Bases/StageEntity.cpp:100 `viewOffset.x << 0xc` (px -> world), and
# src/Bases/Coin.cpp:650 setting viewOffset.y = 8 while :705 offsets the sprite by
# 0x8000 -- 8 px == 0x8000, so 1 px == 0x1000.
FIX = 4096.0


def load(path):
    with open(path, "rb") as f:
        return f.read()


def u32s(blob):
    return struct.unpack_from(f"<{len(blob) // 4}i", blob, 0)


def candidates_pair(a, b, direction="right", window=(0, 60000)):
    """32-bit addresses whose value moved between A and B in the expected direction.

    `window` is in PIXELS, not raw 16.16 units -- converting before comparing is
    the whole point of this function.
    """
    va, vb = u32s(a), u32s(b)
    out = []
    for i in range(len(va)):
        d = vb[i] - va[i]
        if d == 0:
            continue
        if direction == "right" and d <= 0:
            continue
        if direction == "left" and d >= 0:
            continue
        bpx, apx = va[i] / FIX, vb[i] / FIX
        if not (window[0] <= bpx <= window[1] and window[0] <= apx <= window[1]):
            continue
        # plausible as a level coordinate: at most ~one screen of travel per frame
        if abs(d) > 4000 * FIX:
            continue
        out.append({"addr": BASE + 4 * i, "before": va[i], "after": vb[i], "delta": d,
                    "delta_px": d / FIX})
    out.sort(key=lambda r: -abs(r["delta"]))
    return out


def candidates_series(dumps, min_r2=0.995):
    """Per address, fit position ~ a*t^2 + b*t + c and measure the fit.

    A vertical coordinate under constant gravity is an exact parabola (r^2 ~ 1.0).
    Thermostat spikes and animations are not.
    """
    n = len(dumps)
    if n < 8:
        raise SystemExit("series mode needs >= 8 dumps")
    width = min(len(d) for d in dumps) // 4
    ts = [i * 1.0 for i in range(n)]
    # quadratic least squares, 3x3 normal equations solved by hand
    S = [sum(t ** k for t in ts) for k in range(7)]
    M = [[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]]

    def solve(m, v):
        m = [row[:] + [v[i]] for i, row in enumerate(copy.deepcopy(m))]
        for c in range(3):
            piv = max(range(c, 3), key=lambda r: abs(m[r][c]))
            if abs(m[piv][c]) < 1e-12:
                return None
            m[c], m[piv] = m[piv], m[c]
            for r in range(3):
                if r != c:
                    f = m[r][c] / m[c][c]
                    for k in range(c, 4):
                        m[r][k] -= f * m[c][k]
        return [m[i][3] / m[i][i] for i in range(3)]

    results = []
    vals = [u32s(d)[:width] for d in dumps]
    for i in range(width):
        col = [vals[f][i] for f in range(n)]
        if len(set(col)) < 4:
            continue
        y = [v / FIX for v in col]
        # same unit rule: the plausibility band is in pixels, not raw integers
        if not all(-8000.0 < v < 120000.0 for v in y):
            continue
        Sy = [sum(t ** k * y[f] for f, t in enumerate(ts)) for k in range(3)]
        coef = solve(M, Sy)
        if not coef:
            continue
        # solve() returns [constant, linear, quadratic] in that order
        c0, c1, c2 = coef
        pred = [c0 + c1 * t + c2 * t * t for t in ts]
        ss_res = sum((yy - pp) ** 2 for yy, pp in zip(y, pred))
        m = sum(y) / n
        ss_tot = sum((yy - m) ** 2 for yy in y)
        if ss_tot < 1e-9:
            continue
        r2 = 1 - ss_res / ss_tot
        if r2 >= min_r2 and abs(c2) > 1e-4:
            results.append({"addr": BASE + 4 * i, "r2": r2, "accel_px_f2": 2 * c2,
                            "accel_px_f2_16p16": 2 * c2 * FIX, "v0_px_f": c1,
                            "span_px": max(y) - min(y)})
    results.sort(key=lambda r: -r["r2"])
    return results


def write_symbols(cands, path):
    """Write a candidate symbols.json from the best findings."""
    syms = {}
    for tag, c in cands.items():
        if not c:
            continue
        best = c[0]
        syms[tag] = {"addr": best["addr"], "scale": "16.16",
                     "note": f"r2={best.get('r2', '-')} delta={best.get('delta_px', '-')}"}
    json.dump(syms, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return path


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pair"
    files = sys.argv[2:]
    if mode == "pair":
        a, b = (load(f) for f in files[:2])
        c = candidates_pair(a, b)
        print(f"{len(c)} coordinate candidates (positive change => moving right)")
        for r in c[:15]:
            print(f"  0x{r['addr']:08X}  {r['before'] / FIX:>12.4f} -> {r['after'] / FIX:>12.4f}"
                  f"   d={r['delta_px']:+.4f} px/frame")
    else:
        dumps = [load(f) for f in files]
        res = candidates_series(dumps)
        print(f"{len(res)} addresses follow a parabola with r2>=0.995")
        for r in res[:15]:
            print(f"  0x{r['addr']:08X} r2={r['r2']:.6f} accel={r['accel_px_f2']:+.6f} px/f^2"
                  f"  ({r['accel_px_f2_16p16']:+.1f} in 16.16) span={r['span_px']:.1f}px")
