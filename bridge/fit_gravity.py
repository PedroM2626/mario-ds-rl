"""Route (B), step 2: turn the captured arc candidates into a gravity number.

Input: jump_arcs.json from capture_jump.py.

Everything is re-derived here from the saved series rather than trusting what the
capture printed, so a change in arcs.py can be re-tested against the same
measurement without making the user jump again.

What has to be true before a number is stated:
  * several fitted descents agree with each other (one arc is an accident);
  * the velocity cross-check lands on the same value from a different estimator;
  * candidates that agree on gravity are reported as one value, not as N options.
If any of that fails, the output says so and states no number.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arcs

data = json.load(open(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "jump_arcs.json")))
cands = data["candidates"]
frames_by_base = {w["base"]: np.array(w["frames"], dtype=float) for w in data["windows"]}

if not cands:
    raise SystemExit("no candidates in the file -- nothing to fit. Capture with more jumps "
                     "or widen WINDOWS in capture_jump.py.")

print(f"{len(cands)} candidates. Recomputing arcs from the saved series.\n")
rows = []
for c in cands:
    t = frames_by_base[c["window"]]
    y = np.array(c["series"], dtype=float)
    n = min(len(t), len(y))
    sc = arcs.score_series(t[:n], y[:n])
    if not sc:
        continue
    p = c.get("partner")
    rows.append({"addr": c["addr"], "col": c["col"], "sc": sc, "part": p,
                 "part_addr": c.get("partner_addr")})

rows.sort(key=lambda r: (-r["sc"]["arcs_fit"], r["sc"]["accel_iqr"]))
hdr = (f"{'address':>11} {'arcs':>4} {'fit':>4} {'accel px/f^2':>13} {'iqr':>7} "
       f"{'amp px':>7} {'air f':>6} {'minR2':>6}  {'velocity partner':>28}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    s, p = r["sc"], r["part"]
    pv = (f"0x{r['part_addr']:08X} {p['slope']:+.4f} ({p['agree_pct']:+.1f}%)"
          if p else "-")
    print(f"0x{r['addr']:08X} {s['n_arcs']:>4} {s['arcs_fit']:>4} {s['accel_med']:>13.4f} "
          f"{s['accel_iqr']:>7.4f} {s['amp_med']:>7.1f} {s['span_med']:>6.0f} "
          f"{s['r2_des_min']:>6.3f}  {pv:>28}")

# --- consensus across candidates: gravity is one number, so agreeing rows are evidence
acc = np.array([r["sc"]["accel_med"] for r in rows if r["sc"]["arcs_fit"] >= 2])
if not len(acc):
    raise SystemExit("\nno candidate has 2 fitted descents -- no number is claimed.")
ref = np.median(acc)
cluster = acc[np.abs(acc - ref) <= 0.10 * abs(ref)]
print(f"\n{len(acc)} candidates fitted >=2 descents; median {ref:+.4f} px/frame^2; "
      f"{len(cluster)} agree within 10%")

confirmed = [r for r in rows if r["part"] and abs(r["part"]["agree_pct"]) < 10.0
             and r["sc"]["arcs_fit"] >= 2]
if len(cluster) >= 2 and confirmed:
    g = float(np.median(cluster))
    best = confirmed[0]
    air = best["sc"]["span_med"]
    amp = best["sc"]["amp_med"]
    v0 = -g * air / 2.0                      # takeoff speed from a symmetric arc
    print(f"\n>>> GRAVITY = {g:+.4f} px/frame^2 = {g * 3600:+.1f} px/s^2 at 60 fps <<<")
    print(f"    from {len(cluster)} independent candidates and, for "
          f"0x{best['addr']:08X}, from its velocity partner 0x{best['part_addr']:08X} "
          f"({best['part']['slope']:+.4f}, {best['part']['agree_pct']:+.1f}%)")
    print(f"    jump height {amp:.1f} px ({amp / 16.0:.2f} tiles), "
          f"air time {air:.0f} frames ({air / 60.0:.2f} s), "
          f"takeoff speed {v0:+.2f} px/frame")
    print(f"    in tile units: {g / 16.0:+.5f} tiles/frame^2")
    print("\n    Still open: the candidate's identity is inferred from its trajectory. "
          "Confirm with a labelled walk/jump micro-capture (verify_player.py) before "
          "treating it as the player's Y.")
else:
    print("\nNOT CLAIMING A NUMBER.")
    if len(cluster) < 2:
        print("  only one candidate (or none) reached a consensus curvature.")
    if not confirmed:
        print("  no candidate's velocity partner agreed with its position curvature "
              "within 10%; a parabola alone cannot tell a jump from an ease.")
    print("  Re-capture with more jumps, or widen the windows in capture_jump.py.")
