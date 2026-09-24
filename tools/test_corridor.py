import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask
from src.ram_state import RamState
from src.course import Course, TILE

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"

emu = DeSmuME()
emu.open(ROM)
emu.savestate.load_file(STATE)
rs = RamState(emu)
rs.discover_mario()

print("Testing corridor trajectory...")
# Advance Mario quickly to X ~ 1450 px
for step in range(80):
    # Action 4: Run + Jump
    for k in [Keys.KEY_RIGHT, Keys.KEY_X]:
        emu.input.keypad_add_key(keymask(k))
    if step in [18, 19, 20, 21, 45, 46, 47, 48]:
        emu.input.keypad_add_key(keymask(Keys.KEY_A))
    for _ in range(8):
        emu.cycle()
    emu.input.keypad_rm_key(keymask(Keys.KEY_A))
    p = rs.poll()
    mb = p['mario_B']
    x = mb / 4096.0 if mb else 0
    if x > 1400:
        print(f"step {step}: x={x:.1f}px, yvel={p['vel']}")
        break

emu.destroy()
print("Test completed successfully.")
