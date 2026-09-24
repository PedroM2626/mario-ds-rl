"""Rigorous differential RAM search for stage clear flag in EUR NSMB DS.

1. Baseline: Run 150 steps of normal gameplay (walking, jumping, bumping pipes).
   Keep a boolean mask `ever_nonzero` of any byte in the entire 4MB RAM that
   was non-zero at ANY point during normal gameplay.
2. Flagpole: Reset to savestate, place Mario at 3980px, step forward 30 steps
   through flagpole into castle.
   Find bytes that:
   - Were ALWAYS zero during the 150 normal steps
   - Were zero at 3980px before touching flagpole
   - Transition to non-zero when passing the flagpole (x >= 4032px)
   - Remain non-zero while in the castle (x = 4249px)
"""
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
    print("[find_exact_clear_flag] Initializing DeSmuME...")
    emu = DeSmuME()
    emu.open(ROM)
    emu.savestate.load_file(STATE)

    print("[find_exact_clear_flag] Running 150 baseline steps from entrance...")
    ever_nonzero = (dump_ram(emu) != 0)

    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    for step in range(150):
        if step % 25 < 6:
            emu.input.keypad_add_key(keymask(Keys.KEY_B))
        else:
            emu.input.keypad_rm_key(keymask(Keys.KEY_B))
        for _ in range(8):
            emu.cycle()
        if step % 5 == 0:
            d = dump_ram(emu)
            ever_nonzero |= (d != 0)

    print(f"[find_exact_clear_flag] Baseline complete. Never-nonzero bytes: {(~ever_nonzero).sum()}")

    print("[find_exact_clear_flag] Reloading savestate for flagpole sequence...")
    emu.savestate.load_file(STATE)

    target_x_px = 3980
    target_x_val = target_x_px * 4096
    target_x_48 = (target_x_px - 40) * 4096
    for a in [0x021C18EC, 0x021C18F8]:
        emu.memory.write_long(a, target_x_val)
    for a in [0x021C1828, 0x021C1834]:
        emu.memory.write_long(a, target_x_48)
    for a in [0x02098240, 0x020DCFA0]:
        emu.memory.write_long(a, (target_x_px - 100) * 4096)

    d_before = dump_ram(emu)

    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    emu.input.keypad_add_key(keymask(Keys.KEY_Y))

    rs = RamState(emu)
    rs.discover_mario()

    # Step forward 15 steps (reach castle)
    for step in range(15):
        for _ in range(8):
            emu.cycle()
    d_castle_1 = dump_ram(emu)

    # Step forward another 20 steps (deep inside castle)
    for step in range(20):
        for _ in range(8):
            emu.cycle()
    d_castle_2 = dump_ram(emu)

    emu.destroy()

    # Filter:
    # 1. Never non-zero during baseline normal gameplay
    # 2. Zero before flagpole
    # 3. Non-zero in castle_1 and castle_2
    mask = (~ever_nonzero) & (d_before == 0) & (d_castle_1 > 0) & (d_castle_2 > 0)
    candidates = np.nonzero(mask)[0]

    print(f"\n[find_exact_clear_flag] Found {len(candidates)} verified candidate bytes!")
    print("=" * 80)
    for idx in candidates:
        addr = RAM_BASE + int(idx)
        print(f"  0x{addr:08X} (offset +0x{int(idx):06X}): before={d_before[idx]} -> castle_1={d_castle_1[idx]} -> castle_2={d_castle_2[idx]}")


if __name__ == "__main__":
    main()
