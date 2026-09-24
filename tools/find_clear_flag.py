"""Differential RAM search to identify CLEAR_FLAG_ADDR in EUR NSMB DS.

Uses a SINGLE DeSmuME instance (reloading the savestate) to avoid C++ core
double-init access violations.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "src")
from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask
from ram_state import RamState

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
RAM_BASE, RAM_SIZE = 0x02000000, 4 * 1024 * 1024


def dump_ram(emu):
    return np.frombuffer(
        bytes(emu.memory.read(RAM_BASE, RAM_BASE + RAM_SIZE, 1, False)),
        dtype=np.uint8).copy()


def main():
    print("[find_clear_flag] Initializing DeSmuME...")
    emu = DeSmuME()
    emu.open(ROM)
    emu.savestate.load_file(STATE)

    print("[find_clear_flag] Collecting baseline run from entrance (40 steps)...")
    d_base_0 = dump_ram(emu)
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    for _ in range(40):
        for _ in range(8):
            emu.cycle()
    emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))
    d_base_1 = dump_ram(emu)

    print("[find_clear_flag] Resetting to savestate for flagpole experiment...")
    emu.savestate.load_file(STATE)

    target_x_px = 3980
    target_x_val = target_x_px * 4096
    target_x_48_val = (target_x_px - 40) * 4096
    addrs_88 = [0x021C18EC, 0x021C18F8]
    addrs_48 = [0x021C1828, 0x021C1834]
    cam_addrs = [0x02098240, 0x020DCFA0]

    for a in addrs_88:
        emu.memory.write_long(a, target_x_val)
    for a in addrs_48:
        emu.memory.write_long(a, target_x_48_val)
    for a in cam_addrs:
        emu.memory.write_long(a, (target_x_px - 100) * 4096)

    # D0: right before flagpole
    d_flag_0 = dump_ram(emu)

    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    emu.input.keypad_add_key(keymask(Keys.KEY_Y))

    rs = RamState(emu)
    rs.discover_mario()

    for step in range(25):
        for _ in range(8):
            emu.cycle()
        p = rs.poll()
        if step % 5 == 0:
            mb = p['mario_B']
            x = mb / 4096.0 if mb else 0
            print(f"  flagpole step {step}: x={x:.1f}px")

    # D1: after flagpole contact (Mario has reached castle)
    d_flag_1 = dump_ram(emu)

    for step in range(25, 50):
        for _ in range(8):
            emu.cycle()

    # D2: in castle celebration
    d_flag_2 = dump_ram(emu)
    emu.destroy()

    print("[find_clear_flag] Analyzing differential RAM signatures...")
    # Candidate criteria:
    # 1. d_flag_0 == 0
    # 2. d_flag_1 != 0 and d_flag_2 != 0
    # 3. d_base_0 == 0 and d_base_1 == 0 (strictly 0 during regular gameplay)
    mask = (d_flag_0 == 0) & (d_flag_1 > 0) & (d_flag_2 > 0) & (d_base_0 == 0) & (d_base_1 == 0)
    candidates = np.nonzero(mask)[0]
    print(f"[find_clear_flag] Found {len(candidates)} candidate bytes.")

    print("\n--- Top Candidate Memory Addresses ---")
    for idx in candidates[:60]:
        addr = RAM_BASE + int(idx)
        val0 = d_flag_0[idx]
        val1 = d_flag_1[idx]
        val2 = d_flag_2[idx]
        print(f"  0x{addr:08X}: D0={val0} -> D1={val1} -> D2={val2} (base_0={d_base_0[idx]}, base_1={d_base_1[idx]})")


if __name__ == "__main__":
    main()
