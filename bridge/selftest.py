"""Self-test for the candidate-finding methodology: synthetic RAM with physics
planted on purpose, plus the exact distractors that fooled the real capture.

Two things are validated and both have burned time already:
  1. ramdiff's pair/series estimators recover a planted 16.16 parabola's accel.
  2. arcs.py's matched filter finds a planted JUMP (sampled at the ~20 Hz the
     bridge really achieves) and rejects slow eases. The previous detector ranked
     candidates by longest monotone run and therefore kept a 66 px / 5 s ease and
     threw away the arcs -- that is the failure this test exists to prevent.

Run:  python selftest.py
"""
import math
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arcs
import ramdiff

FIX = ramdiff.FIX
BASE = ramdiff.BASE
random.seed(7)

# ---- planted physics: 60 fps game, sampled every 3 frames (what the bridge does)
STEP = 3
NS = 400                       # samples -> 1200 frames -> 20 s
G = -0.1875                    # px/frame^2  <- the gravity to be recovered
V0 = 6.0                       # px/frame upward at takeoff
AIR = int(round(-2 * V0 / G))  # 64 frames
GROUND = 128.0
JUMPS = [120, 360, 600, 840, 1080]   # takeoff frames, 2 s cadence

Y_ADDR, VY_ADDR, Y2_ADDR = 0x2000, 0x2004, 0x2008
RAMP, EASE, JITTER, SAW, SINE, FAST = 0x3000, 0x3008, 0x3010, 0x3018, 0x3020, 0x3028


def jump_curve(f, gnd, v0=V0, a=G):
    """Vertical position at game frame f for a repeated jump; ground otherwise."""
    for t0 in JUMPS:
        tau = f - t0
        if 0 <= tau <= AIR:
            return gnd + v0 * tau + 0.5 * a * tau * tau
    return gnd


def jump_vel(f, v0=V0, a=G):
    for t0 in JUMPS:
        tau = f - t0
        if 0 <= tau <= AIR:
            return v0 + a * tau
    return 0.0


dumps = []
for s in range(NS):
    f = s * STEP
    blob = bytearray(0x10000)
    struct.pack_into("<i", blob, Y_ADDR, int(round(jump_curve(f, GROUND) * FIX)))
    struct.pack_into("<i", blob, VY_ADDR, int(round(jump_vel(f) * FIX)))
    # inverted axis: same motion, but up means the number gets smaller
    struct.pack_into("<i", blob, Y2_ADDR, int(round((400.0 - (jump_curve(f, GROUND) - GROUND)) * FIX)))
    # distractors, each a real false positive seen in live data
    struct.pack_into("<i", blob, RAMP, int(round((60.0 + 1.0 * f) * FIX)))
    ease = 128.0 + 66.0 * (1.0 - math.exp(-0.02 * f))          # the 5 s / 66 px ease
    struct.pack_into("<i", blob, EASE, int(round(ease * FIX)))
    struct.pack_into("<i", blob, JITTER, random.randint(0, 3))  # flag word
    struct.pack_into("<i", blob, SAW, int(round((10.0 * (f % 60) / 6.0) * FIX)))
    struct.pack_into("<i", blob, SINE, int(round((200.0 + 60.0 * math.sin(2 * math.pi * f / 4000.0)) * FIX)))
    struct.pack_into("<i", blob, FAST, int(round(jump_curve(f, GROUND, 4.0, -6.0) * FIX)))
    dumps.append(bytes(blob))

series = {a: np.array([struct.unpack_from("<i", d, a)[0] / FIX for d in dumps])
          for a in (Y_ADDR, VY_ADDR, Y2_ADDR, RAMP, EASE, JITTER, SAW, SINE, FAST)}
frames = np.array([s * STEP for s in range(NS)], dtype=float)
M = np.vstack([np.frombuffer(d, dtype=np.int32).astype(np.int64) for d in dumps])

print("== ramdiff series estimator on a full-rate planted parabola ==")
# Separate dataset: candidates_series fits ONE quadratic to the whole series, so
# it is only a valid tool for a single excursion. Feeding it repeated jumps would
# fail for the right reason, which would tell us nothing about its algebra.
single = []
for f in range(60):
    blob = bytearray(0x10000)
    struct.pack_into("<i", blob, Y_ADDR, int(round((40.0 + V0 * f + 0.5 * G * f * f) * FIX)))
    struct.pack_into("<i", blob, RAMP, int(round((60.0 + 1.0 * f) * FIX)))
    struct.pack_into("<i", blob, JITTER, random.randint(0, 3))
    struct.pack_into("<i", blob, SAW, int(round((9.0 if f % 2 else -9.0) * FIX)))
    single.append(bytes(blob))
res = ramdiff.candidates_series(single)
yhit = [r for r in res if r["addr"] == BASE + Y_ADDR]
assert yhit, "FAILED: planted Y not recovered by ramdiff.candidates_series"
assert abs(yhit[0]["accel_px_f2"] - G) < 1e-4, yhit[0]
for name, addr in (("linear ramp", BASE + RAMP), ("jitter", BASE + JITTER),
                   ("sawtooth", BASE + SAW)):
    assert not any(r["addr"] == addr for r in res), f"FALSE POSITIVE in candidates_series: {name}"
print(f"  accel {yhit[0]['accel_px_f2']:+.6f} px/frame^2 == planted {G:+.6f}  "
      f"r2={yhit[0]['r2']:.8f}; 3 distractors rejected")

print("\n== curvature prescreen (the cheap gate the live scan runs first) ==")
pre = arcs.curvature_prescreen(frames, M)
for addr, nm in ((Y_ADDR, "planted Y"), (Y2_ADDR, "planted Y inverted"),
                 (EASE, "exp ease"), (SAW, "sawtooth"), (SINE, "slow sine"),
                 (JITTER, "jitter"), (FAST, "g=6.0 bounce")):
    c = addr // 4
    print(f"  {nm:<16} maxrun={pre['maxrun'][c]:>3} runs={pre['runs'][c]:>3}")
assert pre["runs"][Y_ADDR // 4] >= 2 and pre["maxrun"][Y_ADDR // 4] >= 4, \
    "FAILED: prescreen threw away the planted jump -- the live scan would never see it"
assert pre["runs"][SAW // 4] == 0, "sawtooth passed the curvature band"
print("  OK: the planted jump survives the prescreen, the sawtooth does not")

print("\n== arcs matched filter at the bridge's real ~20 Hz rate ==")
names = {Y_ADDR: "planted Y (up=+)", VY_ADDR: "planted Vy", Y2_ADDR: "planted Y (up=-)",
         RAMP: "linear ramp", EASE: "exp ease 66px/5s", JITTER: "flag jitter",
         SAW: "sawtooth", SINE: "slow sine", FAST: "g=6.0 bounce"}
scores = {}
for addr, y in series.items():
    sc = arcs.score_series(frames, y)
    scores[addr] = sc
    if sc:
        print(f"  {names[addr]:<20} 0x{addr:04X} arcs={sc['n_arcs']} fit={sc['arcs_fit']} "
              f"accel={sc['accel_med']:+.4f} px/f^2 iqr={sc['accel_iqr']:.4f} "
              f"amp={sc['amp_med']:.1f}px span={sc['span_med']:.0f}f "
              f"minR2desc={sc['r2_des_min']:.4f}")
    else:
        print(f"  {names[addr]:<20} 0x{addr:04X} rejected")

assert scores[Y_ADDR], "FAILED: planted Y not detected by arcs.score_series"
assert abs(scores[Y_ADDR]["accel_med"] - G) <= 0.1 * abs(G), \
    f"gravity off: {scores[Y_ADDR]['accel_med']} vs {G}"
assert scores[Y2_ADDR] and abs(scores[Y2_ADDR]["accel_med"] + G) <= 0.1 * abs(G), \
    "FAILED: inverted axis not handled"
assert scores[Y_ADDR]["arcs_fit"] >= 4, "need most of the 5 planted jumps fitted"
print(f"  OK: {scores[Y_ADDR]['arcs_fit']}/5 descents fitted, "
      f"gravity {scores[Y_ADDR]['accel_med']:+.4f} vs planted {G:+.4f} (within 10%), "
      f"and the inverted-axis copy agrees in magnitude")

for addr in (RAMP, EASE, JITTER, SAW, SINE, FAST):
    assert scores[addr] is None, f"FALSE POSITIVE: {names[addr]} passed the filter"
print("  OK: all six distractors rejected, including the ease that fooled the live scan")

print("\n== independent cross-check: velocity, not shape ==")
cand_col = Y_ADDR // 4
sc = scores[Y_ADDR]
partner = arcs.velocity_partner(frames, M, cand_col, sc["arcs"], sc["accel_med"])
by_col = {p["col"]: p for p in partner}
print(f"  {len(partner)} columns are straight lines over the same descent windows")
for p in partner[:5]:
    print(f"    col 0x{p['addr_off']:04X} n={p['n']} slope={p['slope']:+.5f} px/f^2 "
          f"spread={p['slope_spread']:.5f} r2min={p['r2_min']:.4f} "
          f"vs position {p['agree_pct']:+.1f}%")
assert cand_col + 1 in by_col, "FAILED: planted Vy not identified as the velocity partner"
p = by_col[cand_col + 1]
assert abs(p["agree_pct"]) < 5.0, f"velocity disagrees with position: {p}"
assert p["slope_spread"] < 0.02, f"velocity slope not consistent across jumps: {p}"
print(f"  OK: planted Vy slope {p['slope']:+.5f} px/frame^2 agrees with the position "
      f"curvature {sc['accel_med']:+.5f} ({p['agree_pct']:+.2f}%) over {p['n']} jumps")

print("\nSELFTEST OK: the arc filter finds planted jumps at 20 Hz, recovers gravity "
      "to within 10%, and rejects the eases/ramps/noise that previously passed.")
