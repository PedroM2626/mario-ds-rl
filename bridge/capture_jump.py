"""Route (B), step 1: find the words that move like a jump, without knowing where
the actor lives in RAM.

Uses only the "dump" command (arbitrary base + length), so it never needs a Lua
reload -- reloading the script kills the emulator on this build.

Detector history, because every version here was wrong in its own way:
  * v1 required ">= 6 sign flips". Winners had 272 flips in 360 samples: jitter.
  * v2 scored by longest monotone run. Its "winner" (0x02098210, 66 px) turned
    out to be a 5-second LINEAR excursion -- an ease, not a jump -- because
    ranking by monotone run actively prefers slow eases over 60-frame arcs.
    Reprocessed with the arc filter, none of v2's 15 survivors had a single arc.
  * v3 (here) prescreens for constant non-zero curvature and then fits arcs
    (arcs.py), which is the same path the self-test validates on planted data.

Windows are non-overlapping. v2 used 0x02080000/0x020A0000/0x020C0000 with a
256 KiB length, so consecutive windows overlapped by 128 KiB and the same
address was scored twice against two different minute-long slices of the
capture. It also never looked at 0x021C0000, where the walking diff had hits.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arcs
import nsmb_env

SEC_PER_WINDOW = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
WIN = 0x40000
# The whole ARM9 work RAM, not a guess. The previous 6 windows were chosen from
# where an older scan found motion, and a full round of them measured nothing
# because the game was sitting on the world map -- a null over a guessed subset
# cannot tell "wrong place" from "nothing happened".
WINDOWS = [0x02000000 + i * WIN for i in range(16)]
if len(sys.argv) > 2:                 # optional: how many windows to sweep (smoke tests)
    WINDOWS = WINDOWS[: int(sys.argv[2])]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TOP_PARTNER_CHECKS = 3          # velocity cross-check costs a pass over the columns


def main():
    b = nsmb_env.Bridge()
    st = b.status()
    print("[status]", st)
    if st.get("emulating") != 1:
        raise SystemExit("ABORTED: game is paused or in a menu -- nothing would be measured.")

    total = int(SEC_PER_WINDOW * len(WINDOWS))
    print(f"\n>>> ENTER A STAGE (not the world map -- nothing jumps there) and JUMP "
          f"REPEATEDLY. Each round sweeps {len(WINDOWS)} windows x {SEC_PER_WINDOW:.0f}s "
          f"= {total}s, and keeps retrying for {ROUNDS} rounds. <<<\n")

    out = os.path.join(HERE, "jump_arcs.json")
    for rnd in range(ROUNDS):
        report, windows_meta = [], []
        if ROUNDS > 1:
            print(f"===== round {rnd + 1}/{ROUNDS} =====")
        for wi, base in enumerate(WINDOWS):
            print(f"[sweep {wi + 1}/{len(WINDOWS)}] 0x{base:08X} +0x{WIN:X}, {SEC_PER_WINDOW:.0f}s")
            frames, rows, t_end = [], [], time.time() + SEC_PER_WINDOW
            i = 0
            while time.time() < t_end:
                try:
                    blob = b.dump(f"j{base:X}_{i}", base, WIN)
                except nsmb_env.BridgeError as e:
                    print("  dump failed:", e)
                    break
                rows.append(np.frombuffer(blob, dtype="<i4"))
                f_now = int(b.status()["framecount"])
                if frames and f_now <= frames[-1]:
                    raise SystemExit(f"ABORTED: framecount went {frames[-1]} -> {f_now}: "
                                     "the game is paused or the ROM was reset mid-capture.")
                frames.append(f_now)
                i += 1
            if len(frames) < 12:
                print(f"  only {len(frames)} samples -- rate too low, window skipped")
                continue

            fr = np.array(frames, dtype=float)
            M = np.vstack(rows)
            span = fr[-1] - fr[0]
            hz = len(fr) / (span / 60.0)
            print(f"  {len(fr)} samples over {span} frames ({hz:.1f} Hz effective)")
            windows_meta.append({"base": base, "frames": fr.tolist()})

            # Positive control: a null from the arc filter only means something if
            # memory was moving at all here. Without it "0 candidates" over the world
            # map looked like a wrong-window conclusion.
            moved = int(np.count_nonzero(M.max(axis=0) != M.min(axis=0)))
            print(f"  activity: {moved} of {M.shape[1]} words changed at all"
                  f"{'  <-- NOTHING MOVES IN THIS WINDOW' if moved < 50 else ''}")

            pre = arcs.curvature_prescreen(fr, M)
            idx = np.nonzero((pre["runs"] >= 2) & (pre["maxrun"] >= 4))[0]
            print(f"  prescreen: {len(idx)} of {M.shape[1]} columns show repeated constant curvature")

            hits = []
            for k in idx:
                sc = arcs.score_series(fr, M[:, int(k)] / arcs.FIX)
                if sc:
                    hits.append((int(k), sc))
            hits.sort(key=lambda h: (-h[1]["arcs_fit"], h[1]["accel_iqr"]))
            print(f"  arc filter: {len(hits)} columns have >=2 fitted parabolic arcs")

            for rank, (k, sc) in enumerate(hits):
                rec = {"addr": int(base + 4 * k), "window": base, "col": k, **sc}
                rec.pop("arcs")
                rec["arcs_detail"] = [{kk: vv for kk, vv in a.items()} for a in sc["arcs"]]
                if rank < TOP_PARTNER_CHECKS:
                    part = arcs.velocity_partner(fr, M, k, sc["arcs"], sc["accel_med"],
                                                 lo=k - 128, hi=k + 128)
                    rec["partner"] = part[0] if part else None
                    rec["partner_addr"] = int(base + 4 * part[0]["col"]) if part else None
                rec["series"] = (M[:, k] / arcs.FIX).round(4).tolist()
                report.append(rec)
                if rank < 6:
                    p = rec.get("partner")
                    print(f"    0x{rec['addr']:08X} arcs={sc['n_arcs']} fit={sc['arcs_fit']} "
                          f"accel={sc['accel_med']:+.4f} iqr={sc['accel_iqr']:.4f} "
                          f"amp={sc['amp_med']:.1f}px span={sc['span_med']:.0f}f "
                          f"partner={'0x%X@%+.4f(%+.1f%%)' % (base + 4 * p['col'], p['slope'], p['agree_pct']) if p else '-'}")
            del M, rows

        json.dump({"windows": windows_meta, "candidates": report}, open(out, "w"), indent=1)
        print(f"\nround {rnd + 1}: {len(report)} candidates saved to {out}")
        if report:
            break
        if rnd + 1 < ROUNDS:
            print("  none: the player's Y is outside these windows, or no jump happened "
                  "during the sweep (world map and menus produce nothing). Retrying.")

    if not report:
        print("Widen WINDOWS (the whole ARM9 heap is 0x02000000..0x02400000) and capture again.")


if __name__ == "__main__":
    main()
