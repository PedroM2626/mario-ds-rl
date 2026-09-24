import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.course import Course, TILE
import json

c = Course(rom_path='data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds')
print(f"Total sprites: {len(c.sprites)}")
print(f"Total rects: {len(c.rects)}")
print(f"Total entrances: {len(c.entrances)}")

print("\n--- SPRITES (Ordered by X position) ---")
for s in sorted(c.sprites, key=lambda s: s['tx']):
    print(f"Col {s['tx']:3d} ({s['x_px']:4d}px, y={s['y_px']:3d}px): type {s['type']:3d} | name: {s['name']}")

print("\n--- PITS ---")
cols = sorted(c.ground_cover.keys())
pits = []
col = cols[0]
while col <= cols[-1]:
    if col not in c.ground_cover:
        start = col
        while col <= cols[-1] and col not in c.ground_cover:
            col += 1
        pits.append((start, col - 1, (col - start) * TILE))
    col += 1
for p in pits:
    print(f"Pit: cols {p[0]}-{p[1]} (x={p[0]*TILE}-{p[1]*TILE+16}px, width={p[2]}px)")

print("\n--- ELEVATION / STAIRS / PIPES ---")
for col in range(cols[0], cols[-1] + 1):
    fl = c.floor_row(col)
    if fl is not None and fl < 28:
        print(f"Elevated floor at col {col:3d} (x={col*TILE:4d}px): floor_row={fl} (height={30-fl} tiles)")
