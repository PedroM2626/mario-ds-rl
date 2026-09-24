"""Jump-arc matched filter, shared by the RAM scan and the gravity fit.

Why this exists: ranking candidate addresses by "longest monotone run" selected
slow easing curves (camera, platform) over real jump arcs, so it produced a
66 px "jump" lasting 5 seconds of game time with a linear descent. Position
versus time for a jump is a parabola, so that is what has to be matched.

Everything is measured against the EMULATOR framecount, not sample index: the
sampler runs ~20 Hz with jitter while the game runs 60 fps, so 3 frames per
sample is normal and index spacing would be wrong.

Convention: y is in pixels (raw RAM value divided by FIX) and t is in game
frames, so a fitted y = a*t^2 + b*t + c gives accel = 2a in px/frame^2.

FIX is 0x1000 per pixel, not 0x10000: see the evidence recorded in ramdiff.FIX.
Using 65536 here shrinks every pixel figure 16-fold, which is how a 66 px "jump"
got reported when the underlying excursion was 66 tiles.
"""
import numpy as np

FIX = 4096.0

CFG = {
    "half_f": 90.0,        # hard cap: frames from the apex before the arc is cut
    "max_side": 14,        # hard cap on samples per side of the apex
    "min_side": 3,         # need this many samples on each side to fit at all
    "min_n_full": 8,
    "min_r2": 0.95,
    "a_min": 0.02,         # px/frame^2. Below this it is an ease, not a fall.
    "a_max": 2.0,          # above this it is a bounce/glitch, not Mario
    "amp_min": 8.0,        # px above the arc's own endpoints
    "max_resid": 4.0,      # px
    "apex_frac": (0.15, 0.85),   # apex must sit inside this band of the window
    "edge_eps_px": 0.05,   # smaller than this counts as "back on the ground"
    "patience": 1,         # allowed non-descending samples before the arc ends
}


def smooth(y, k=3):
    """Box filter, edge-replicative. Only used to pick the apex, never to fit."""
    if k <= 1 or len(y) < k:
        return y
    pad = k // 2
    z = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(z, np.ones(k) / k, mode="valid")[: len(y)]


def fit_poly(t, y, deg=2):
    """Least-squares fit returning (coefs highest-first, r2, rms residual)."""
    A = np.vstack([t ** p for p in range(deg, -1, -1)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else -1.0
    return coef, r2, float(np.sqrt(ss_res / len(t)))


def _arc_edges(t, y, i, sign, cfg):
    """Walk outward from the apex until the series stops moving away from it.

    A fixed frame window is what broke the first version: it bled into the ground
    plateau on both sides, so the quadratic fit averaged airborne samples with
    flat ones, dropping the curvature from -0.1875 to -0.1645 and pushing the
    residual past the gate. Bounding the window by the motion itself keeps the
    fit on the airborne part, and it also kills takeoff/landing corner
    "extrema", which have no samples on one side.
    """
    eps, pat = cfg["edge_eps_px"], cfg["patience"]
    edges = []
    for step in (-1, 1):
        j, miss = i, 0
        while True:
            k = j + step
            if k < 0 or k >= len(y) or abs(t[k] - t[i]) > cfg["half_f"]:
                break
            if sign * (y[k] - y[j]) < -eps:
                j, miss = k, 0
            else:
                miss += 1
                if miss > pat:
                    break
        edges.append(j)
    lo, hi = edges
    return max(i - cfg["max_side"], lo), min(i + cfg["max_side"] + 1, hi + 1)


def find_arcs(t, y, cfg=CFG):
    """Locate parabolic arcs in one series and return the fits that describe each.

    A held jump is NOT free fall: while the button is down the engine applies its
    own upward acceleration and there is an intentional hang near the apex. So
    each arc is fitted three ways (whole arc, ascent only, descent only) and a
    gravity number may only ever come from the descent half.
    """
    out = []
    n = len(y)
    if n < cfg["min_n_full"] + 2:
        return out
    s = smooth(y, 3)
    for i in range(1, n - 1):
        is_max = s[i] >= s[i - 1] and s[i] > s[i + 1]
        is_min = s[i] <= s[i - 1] and s[i] < s[i + 1]
        if not (is_max or is_min):
            continue
        sign = 1 if is_max else -1
        lo, hi = _arc_edges(t, y, i, sign, cfg)
        # Trim samples that already sit on the plateau the walk stopped at. The
        # landing sample equals the ground value while the parabola would have put
        # him below it; keeping it flattens the curvature (it cost 20% of the
        # planted gravity in the self-test).
        eps = cfg["edge_eps_px"]
        base_l = y[lo - 1] if lo > 0 else y[lo]
        while lo < i and abs(y[lo] - base_l) <= eps:
            lo += 1
        base_r = y[hi] if hi < n else y[hi - 1]
        while hi - 1 > i and abs(y[hi - 1] - base_r) <= eps:
            hi -= 1
        if i - lo < cfg["min_side"] or hi - 1 - i < cfg["min_side"]:
            continue

        tw, yw = t[lo:hi], y[lo:hi]
        # Fit in the raw axis so the reported sign is the physical one: gravity is
        # negative when Y grows upward and positive when Y grows downward. The
        # only thing the polarity decides is which way the parabola must open.
        coef, r2, resid = fit_poly(tw - tw[0], yw, 2)
        a = coef[0]
        if sign * a >= 0:                 # a peak must open downward, a trough upward
            continue
        span = tw[-1] - tw[0]
        apex = -coef[1] / (2 * coef[0]) if coef[0] != 0 else np.nan
        lo_f, hi_f = cfg["apex_frac"]
        amp = sign * (y[i] - 0.5 * (yw[0] + yw[-1]))
        rec = {
            "apex_i": i, "apex_f": float(t[i]), "lo": lo, "hi": hi,
            "n": len(tw), "span_f": float(span),
            "accel_full": float(2 * a), "r2_full": float(r2), "resid_full": float(resid),
            "amp_px": float(amp), "polarity": "up" if is_max else "down",
            "apex_frac": float(apex / span) if np.isfinite(apex) and span else float("nan"),
        }
        for lab, (ts, ys) in (("ascent", (tw[: i - lo + 1], yw[: i - lo + 1])),
                              ("descent", (tw[i - lo:], yw[i - lo:]))):
            if len(ts) >= cfg["min_side"] + 2:
                c2, r2b, resb = fit_poly(ts - ts[0], ys, 2)
                rec[f"accel_{lab}"] = float(2 * c2[0])
                rec[f"r2_{lab}"] = float(r2b)
                rec[f"resid_{lab}"] = float(resb)
        out.append(rec)
    return out


def arc_is_plausible(rec, cfg=CFG):
    """Gate one arc fit: real curvature, real parabola, apex inside the window."""
    a = abs(rec["accel_full"])
    return (rec["n"] >= cfg["min_n_full"] and rec["r2_full"] >= cfg["min_r2"]
            and cfg["a_min"] <= a <= cfg["a_max"]
            and rec["amp_px"] >= cfg["amp_min"]
            and rec["resid_full"] <= cfg["max_resid"]
            and cfg["apex_frac"][0] <= rec["apex_frac"] <= cfg["apex_frac"][1])


def score_series(t, y, cfg=CFG):
    """Summarise a series by the plausible arcs it contains. None if it has < 2."""
    arcs = [r for r in find_arcs(t, y, cfg) if arc_is_plausible(r, cfg)]
    if len(arcs) < 2:
        return None
    # collapse arcs whose windows overlap: one jump can register at 2 adjacent samples
    keep = []
    for r in sorted(arcs, key=lambda x: x["apex_i"]):
        if keep and r["lo"] < keep[-1]["hi"]:
            if r["r2_full"] > keep[-1]["r2_full"]:
                keep[-1] = r
            continue
        keep.append(r)
    acc = np.array([r["accel_descent"] for r in keep
                    if "accel_descent" in r and r["r2_descent"] >= cfg["min_r2"]])
    return {"n_arcs": len(keep), "arcs": keep,
            "arcs_fit": len(acc),
            "accel_med": float(np.median(acc)) if len(acc) else float("nan"),
            "accel_iqr": float(np.subtract(*np.percentile(acc, [75, 25]))) if len(acc) > 2 else 0.0,
            "r2_des_min": float(min(r.get("r2_descent", -1) for r in keep)),
            "amp_med": float(np.median([r["amp_px"] for r in keep])),
            "span_med": float(np.median([r["span_f"] for r in keep]))}


def curvature_prescreen(t, M, cfg=CFG, min_run=4, min_runs=2, block=4096):
    """Cheap vectorised gate: which columns show constant, non-zero curvature?

    The second difference of position IS curvature, and free fall has a constant
    one, so this throws away everything that cannot be a jump without fitting a
    single quadratic. It is deliberately permissive -- an ease toward a target
    passes it -- because the decision is made by find_arcs plus the velocity
    cross-check. Its only job is to keep 65536 columns per window tractable.

    Returns a dict of numpy arrays indexed by column: maxrun, runs.
    """
    t = np.asarray(t, dtype=np.float32)
    n, ncols = M.shape
    dt = np.diff(t)
    if len(dt) < min_run + 2:
        return {"maxrun": np.zeros(ncols, dtype=np.int32), "runs": np.zeros(ncols, dtype=np.int32)}
    best_run = np.zeros(ncols, dtype=np.int32)
    best_cnt = np.zeros(ncols, dtype=np.int32)
    for c0 in range(0, ncols, block):
        c1 = min(c0 + block, ncols)
        px = M[:, c0:c1].astype(np.float32) / FIX
        V = np.diff(px, axis=0) / dt[:, None]
        A = np.diff(V, axis=0) / dt[1:, None]
        cnt = np.zeros(c1 - c0, dtype=np.int32)
        mx = np.zeros(c1 - c0, dtype=np.int32)
        for s in (1.0, -1.0):
            m = (A * s >= cfg["a_min"]) & (A * s <= cfg["a_max"])
            r = np.zeros(m.shape, dtype=np.int16)
            for i in range(1, m.shape[0]):
                r[i] = np.where(m[i], r[i - 1] + 1, 0)
            hit = (r == min_run)          # counts each run once, at its 4th sample
            cnt += hit.sum(axis=0)
            mx = np.maximum(mx, r.max(axis=0))
        best_run[c0:c1] = mx
        best_cnt[c0:c1] = cnt
    return {"maxrun": best_run, "runs": best_cnt}


def velocity_partner(t, M, cand_col, arcs, accel_px_f2, cfg=CFG, lo=None, hi=None):
    """Independent check: differentiate the arc instead of trusting its shape.

    dV/dt is the acceleration, so if one column really is the vertical position
    with curvature `accel_px_f2`, the vertical velocity column must be a straight
    line of slope `accel_px_f2` over the very same descent windows. Two different
    estimators landing on one number is the evidence -- a parabola alone is not,
    because an ease toward a target also looks like one.

    lo/hi bound the column range. In the live scan they are the candidate's own
    neighbourhood: engine structs keep position and velocity adjacent, and a
    match 1 MiB away would be coincidence rather than evidence.
    """
    hits = []
    lo = 0 if lo is None else max(0, lo)
    hi = M.shape[1] if hi is None else min(M.shape[1], hi)
    for col in range(lo, hi):
        if col == cand_col:
            continue
        slopes, r2s = [], []
        for r in arcs:
            a_idx, b_idx = r["apex_i"], r["hi"]
            if b_idx - a_idx < cfg["min_side"] + 2:
                continue
            ts, ys = t[a_idx:b_idx], M[a_idx:b_idx, col] / FIX
            c1, rr, _ = fit_poly(ts - ts[0], ys, 1)
            slopes.append(c1[0])
            r2s.append(rr)
        if len(slopes) < max(2, len(arcs) // 2):
            continue
        if min(r2s) < cfg["min_r2"]:          # not a straight line: not a velocity
            continue
        m = float(np.mean(slopes))
        spread = float(np.max(slopes) - np.min(slopes))
        hits.append({"col": col, "addr_off": col * 4, "n": len(slopes),
                     "slope": m, "slope_spread": spread,
                     "r2_min": float(min(r2s)),
                     "agree_pct": 100.0 * (m - accel_px_f2) / accel_px_f2
                     if accel_px_f2 else float("nan")})
    hits.sort(key=lambda h: abs(h["slope"] - accel_px_f2))
    return hits
