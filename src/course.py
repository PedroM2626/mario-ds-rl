"""Direct ROM parser for NSMB DS level files (validated for EUR 1-1).

course/X##_#.bin : header 14x(off,len) + blocks (sprites in Blocks[6], 12B:
                    type u16, x u16, y u16 (TILES), data 6B; entrances in
                    Blocks[5], 20B: x,y,cam... (PIXELS)).
course/X##_#_bgdat.bin : tile objects (obj u8, tileset u8, x,y,w,h u16 in
                    TILES, end 0xFFFF). obj 0x0A/0x00 = ground.
Calibrations validated for A01_1 (1-1): spawn tile 3 (X_RAM 48px / B 88px =
                    entrance (80,464) + half-sprite height), ground at row
                    30 (Y_RAM -480px), goomba at tile 24 (RAM tracked at 365px).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TILE = 16
# All solid surface objects from tileset (ground, grass, slopes, edges, stairs, pipes)
# 0x0A: ground, 0x00: base, 0x09: grass top, 0x06-0x08: slopes/edges, 0x0D: stairs,
GROUND_OBJS = (0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0D, 0x0E, 0x0F, 0x2C, 0x30)

import json

# course sprite IDs (NSMBe) -> names; RAM actor IDs differ (e.g. goomba: 148 vs 0xA0)
SPRITE_NAMES = {148: "Goomba", 149: "Koopa", 117: "?", 132: "?", 45: "?",
                198: "?", 199: "?", 32: "Flagpole", 264: "?", 155: "?", 235: "?"}

def _load_extended_sprite_names():
    names = dict(SPRITE_NAMES)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "nsmb_object_ids.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.get("id_to_classes", {}).items():
                    ik = int(k)
                    if ik not in names or names[ik] == "?":
                        names[ik] = v[0] if isinstance(v, list) and v else str(v)
        except Exception:
            pass
    return names

SPRITE_NAMES = _load_extended_sprite_names()


class Course:
    def __init__(self, name="course/A01_1.bin", rom_path=None):
        import ndspy.rom
        if rom_path is None:
            rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data",
                                    "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
        rom = ndspy.rom.NintendoDSRom.fromFile(rom_path)

        raw = bytes(rom.getFileByName(name))
        assert raw is not None, f"{name} not found in ROM"
        blocks = [struct.unpack_from("<II", raw, i * 8) for i in range(14)]
        get = lambda n: raw[blocks[n][0]:blocks[n][0] + blocks[n][1]]

        # sprites: Blocks[6], 12B (type,x,y tiles, data 6B)
        # Structurally proven: 12-byte records from offset 0, ended by 0xFFFFFFFF u32 trailer
        seg = get(6)
        self.sprites = []
        n_records = (len(seg) - 4) // 12 if len(seg) >= 4 and (len(seg) - 4) % 12 == 0 else len(seg) // 12
        for i in range(max(0, n_records)):
            t, x, y = struct.unpack_from("<HHH", seg, i * 12)
            self.sprites.append({"type": t, "tx": x, "ty": y,
                                 "x_px": x * TILE, "y_px": y * TILE,
                                 "name": SPRITE_NAMES.get(t, f"id{t}")})
        # entrances: Blocks[5], 20B (x,y pixels)
        seg = get(5)
        self.entrances = []
        for i in range(len(seg) // 20):
            x, y = struct.unpack_from("<HH", seg, i * 20)
            self.entrances.append({"x_px": x, "y_px": y})

        # bgdat: tile objects
        bg = bytes(rom.getFileByName(name.replace(".bin", "_bgdat.bin")))
        assert bg is not None
        self.rects = []
        i = 0
        while i + 10 <= len(bg):
            obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", bg, i)
            if obj == 0xFF:
                break
            self.rects.append({"obj": obj, "tx": x, "ty": y, "w": w, "h": h})
            i += 10

        # ground coverage per column (tiles) - filter yy >= 18 to exclude sky/ceiling markers at row 0
        cover = {}
        for r in self.rects:
            if r["obj"] in GROUND_OBJS:
                for xx in range(r["tx"], r["tx"] + r["w"]):
                    for yy in range(r["ty"], r["ty"] + r["h"]):
                        if yy >= 18:
                            cover.setdefault(xx, set()).add(yy)
        self.ground_cover = cover

        # flag / level finish detection: sprite 32 or highest X goal object
        flag_candidates = [s for s in self.sprites if s["type"] == 32 or "Flag" in s["name"]]
        if flag_candidates:
            self.flag_x_px = float(min(s["x_px"] for s in flag_candidates))
        else:
            self.flag_x_px = 4032.0

    def floor_row(self, tile_x):
        """Ground row at column or None if pit."""
        rows = self.ground_cover.get(int(tile_x))
        return min(rows) if rows else None

    def pit_ahead(self, mario_px, n=6):
        """List of n flags (1.0 solid / 0.0 pit) for columns ahead."""
        col0 = int(mario_px // TILE)
        return [1.0 if self.floor_row(col0 + m) is not None else 0.0
                for m in range(1, n + 1)]

    def goomba_spawns_px(self):
        return [s["x_px"] for s in self.sprites if s["type"] == 148]


if __name__ == "__main__":
    c = Course()
    print(f"sprites: {len(c.sprites)}, rects: {len(c.rects)}, "
          f"entrances: {c.entrances[:1]}", flush=True)
    # Cross-validations with RAM
    e = c.entrances[0]
    assert (e["x_px"], e["y_px"]) == (80, 464), e
    assert c.floor_row(3) == 30 and c.floor_row(30) in (29, 30), f"row30={c.floor_row(30)}"
    assert c.flag_x_px == 4032.0, f"flag_x_px={c.flag_x_px}"
    gs = c.goomba_spawns_px()
    assert 24 * TILE in gs, gs[:5]
    print(f"OK: spawn={e}, ground tile3 row={c.floor_row(3)}, "
          f"tile30 row={c.floor_row(30)}, flag_x={c.flag_x_px}, goombas px={gs[:6]}", flush=True)
    print("pit_ahead(48px,6):", c.pit_ahead(48.0), flush=True)
    print("pit_ahead(400px,6):", c.pit_ahead(400.0), flush=True)
