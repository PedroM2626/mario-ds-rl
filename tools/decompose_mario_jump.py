"""Empirical Jump State Machine Decomposition for Mario in NSMB DS (EUR release).

Reverse Engineering Route B:
Executes high-frequency 60.0 Hz (single-frame) telemetry captures on DeSmuME
to isolate exact vertical launch velocities, piecewise gravities (hold vs fall),
apex hang times, terminal velocities, and horizontal momentum scaling across:
  1. Standing tap-jump (short hop)
  2. Standing full held-jump
  3. Walking held-jump
  4. Running (dashing) held-jump
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "src")
from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
MARIO_BASE = 0x021C1890
OFF_A, OFF_B = 0x60, 0x68
YVEL_ADDR = 0x021C1908
XVEL_ADDR = 0x021C1904


def _s32(val):
    return val - 2**32 if val >= 2**31 else val


def capture_jump_trial(emu, setup_keys, jump_keys, jump_hold_frames, total_frames=120):
    """Executes a single jump trial at 60 Hz and records trajectory."""
    emu.savestate.load_file(STATE)

    # Let physics settle on ground for 10 frames
    for _ in range(10):
        emu.cycle()

    # Apply setup (e.g. running forward to build horizontal speed)
    for k in setup_keys:
        emu.input.keypad_add_key(keymask(k))
    for _ in range(25):
        emu.cycle()

    # Now press jump keys
    for k in jump_keys:
        emu.input.keypad_add_key(keymask(k))

    frames = []
    x_px_list, y_px_list, vx_list, vy_list = [], [], [], []

    for f in range(total_frames):
        if f == jump_hold_frames:
            for k in jump_keys:
                emu.input.keypad_rm_key(keymask(k))

        emu.cycle()

        ma = _s32(emu.memory.read(MARIO_BASE + OFF_A, MARIO_BASE + OFF_A, 4, False))
        mb = _s32(emu.memory.read(MARIO_BASE + OFF_B, MARIO_BASE + OFF_B, 4, False))
        yv = _s32(emu.memory.read(YVEL_ADDR, YVEL_ADDR, 4, False))
        xv = _s32(emu.memory.read(XVEL_ADDR, XVEL_ADDR, 4, False))

        frames.append(f)
        x_px_list.append(mb / 4096.0)
        y_px_list.append(ma / 4096.0)
        vx_list.append(xv / 4096.0)
        vy_list.append(yv / 4096.0)

    # Release any remaining keys
    for k in setup_keys:
        emu.input.keypad_rm_key(keymask(k))

    return {
        "frames": np.array(frames),
        "x": np.array(x_px_list),
        "y": np.array(y_px_list),
        "vx": np.array(vx_list),
        "vy": np.array(vy_list),
    }


def analyze_jump_trajectory(traj, name):
    y = traj["y"]
    vy = traj["vy"]
    vx = traj["vx"]
    x = traj["x"]

    # Floor baseline is initial y
    y_floor = y[0]
    dy = y - y_floor  # in pixels (positive = upward)

    # Airborne mask: where Mario is above floor by at least 0.5 px or vy != 0
    airborne = np.nonzero(dy > 0.5)[0]
    if len(airborne) == 0:
        return {"name": name, "airborne": False}

    t_takeoff = airborne[0]
    t_land = airborne[-1]
    air_duration = int(t_land - t_takeoff + 1)

    peak_idx = int(np.argmax(dy))
    h_max = float(dy[peak_idx])

    # Initial vertical launch velocity
    vy_launch = float(vy[t_takeoff])

    # Ascent phase: from takeoff to peak
    ascent_f = np.arange(t_takeoff, peak_idx)
    if len(ascent_f) > 2:
        ascent_vy = vy[ascent_f]
        # Linear fit on vy to find gravity: vy(t) = vy_0 - g_ascent * t
        p_asc = np.polyfit(ascent_f - t_takeoff, ascent_vy, deg=1)
        g_ascent = float(-p_asc[0])  # acceleration magnitude (px/frame^2)
    else:
        g_ascent = float("nan")

    # Descent phase: from peak to landing
    descent_f = np.arange(peak_idx, t_land)
    if len(descent_f) > 2:
        descent_vy = vy[descent_f]
        p_desc = np.polyfit(descent_f - peak_idx, descent_vy, deg=1)
        g_descent = float(-p_desc[0])
    else:
        g_descent = float("nan")

    # Terminal vertical velocity (maximum downward speed during fall)
    vy_terminal = float(np.min(vy[t_takeoff:t_land + 1]))

    # Apex hang time: frames near peak where |vy| < 0.5 px/frame
    apex_frames = int(np.sum((np.abs(vy[t_takeoff:t_land + 1]) < 0.5)))

    # Horizontal kinematics
    vx_mean = float(np.mean(vx[t_takeoff:t_land + 1]))
    x_reach = float(x[t_land] - x[t_takeoff])

    return {
        "name": name,
        "airborne": True,
        "t_takeoff_frame": int(t_takeoff),
        "t_land_frame": int(t_land),
        "air_duration_frames": air_duration,
        "max_height_px": round(h_max, 2),
        "vy_launch_px_per_frame": round(vy_launch, 4),
        "gravity_ascent_px_per_f2": round(g_ascent, 4),
        "gravity_descent_px_per_f2": round(g_descent, 4),
        "vy_terminal_px_per_frame": round(vy_terminal, 4),
        "apex_hang_frames": apex_frames,
        "vx_mean_px_per_frame": round(vx_mean, 4),
        "horizontal_reach_px": round(x_reach, 2),
    }


def main():
    print("[decompose_mario_jump] Initializing DeSmuME...")
    emu = DeSmuME()
    emu.open(ROM)

    trials = [
        ("Standing Tap Jump (2 frames)", [], [Keys.KEY_B], 2),
        ("Standing Full Jump (Hold B)", [], [Keys.KEY_B], 40),
        ("Walking Held Jump (Right + B)", [Keys.KEY_RIGHT], [Keys.KEY_B], 40),
        ("Running Held Leap (Right + Y + B)", [Keys.KEY_RIGHT, Keys.KEY_Y], [Keys.KEY_B], 48),
    ]

    results = {}
    for name, setup, jump, hold in trials:
        print(f"[decompose_mario_jump] Running trial: {name}...")
        traj = capture_jump_trial(emu, setup, jump, hold, total_frames=90)
        res = analyze_jump_trajectory(traj, name)
        results[name] = res
        print(f"  -> Duration: {res.get('air_duration_frames')}f | Max H: {res.get('max_height_px')}px | "
              f"Reach: {res.get('horizontal_reach_px')}px | Launch Vy: {res.get('vy_launch_px_per_frame')} | "
              f"g_ascent: {res.get('gravity_ascent_px_per_f2')} | g_descent: {res.get('gravity_descent_px_per_f2')}")

    emu.destroy()

    out_path = "out/dataset/physics_jump_state_machine.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[decompose_mario_jump] Successfully saved decomposition to {out_path}!")


if __name__ == "__main__":
    main()
