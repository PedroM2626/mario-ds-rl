"""Parser de fases NSMB DS direto da ROM (validado p/ 1-1 EUR).

course/X##_#.bin : header 14x(off,len) + blocos (sprites em Blocks[6], 12B:
                    type u16, x u16, y u16 (TILES), data 6B; entrances em
                    Blocks[5], 20B: x,y,cam... (PIXELS)).
course/X##_#_bgdat.bin : objetos de tile (obj u8, tileset u8, x,y,w,h u16 em
                    TILES, fim 0xFFFF). obj 0x0A/0x00 = chao.
Calibragens validadas p/ A01_1 (1-1): spawn tile 3 (X_RAM 48px / B 88px =
                    entrada (80,464) + meia-altura do sprite), chao na fileira
                    30 (Y_RAM -480px), goomba no tile 24 (RAM o viu em 365px).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TILE = 16
GROUND_OBJS = (0x0A, 0x00)

# course sprite IDs (NSMBe) -> nomes; RAM actor IDs diferem (ex. goomba: 148 vs 0xA0)
SPRITE_NAMES = {148: "Goomba", 149: "Koopa", 117: "?", 132: "?", 45: "?",
                198: "?", 199: "?", 32: "?", 264: "?", 155: "?", 235: "?"}


class Course:
    def __init__(self, name="course/A01_1.bin", rom_path=None):
        import ndspy.rom
        if rom_path is None:
            rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data",
                                    "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
        rom = ndspy.rom.NintendoDSRom.fromFile(rom_path)

        raw = bytes(rom.getFileByName(name))
        assert raw is not None, f"{name} nao achado na ROM"
        blocks = [struct.unpack_from("<II", raw, i * 8) for i in range(14)]
        get = lambda n: raw[blocks[n][0]:blocks[n][0] + blocks[n][1]]

        # sprites: Blocks[6], 12B (type,x,y tiles, data 6B)
        seg = get(6)
        self.sprites = []
        for i in range((len(seg) - 2) // 12):
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

        # bgdat: objetos de tile
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

        # cobertura do chao por coluna (tiles)
        cover = {}
        for r in self.rects:
            if r["obj"] in GROUND_OBJS:
                for xx in range(r["tx"], r["tx"] + r["w"]):
                    for yy in range(r["ty"], r["ty"] + r["h"]):
                        cover.setdefault(xx, set()).add(yy)
        self.ground_cover = cover

    def floor_row(self, tile_x):
        """Fileira do chao na coluna ou None (pit)."""
        rows = self.ground_cover.get(int(tile_x))
        return min(rows) if rows else None

    def pit_ahead(self, mario_px, n=6):
        """Lista n flags (1.0 solido / 0.0 pit) p/ colunas a frente."""
        col0 = int(mario_px // TILE)
        return [1.0 if self.floor_row(col0 + m) is not None else 0.0
                for m in range(1, n + 1)]

    def goomba_spawns_px(self):
        return [s["x_px"] for s in self.sprites if s["type"] == 148]


if __name__ == "__main__":
    c = Course()
    print(f"sprites: {len(c.sprites)}, rects: {len(c.rects)}, "
          f"entrances: {c.entrances[:1]}", flush=True)
    # validacoes cruzadas com RAM
    e = c.entrances[0]
    assert (e["x_px"], e["y_px"]) == (80, 464), e
    assert c.floor_row(3) == 30 and c.floor_row(30) is None, "pit tile 30?"
    gs = c.goomba_spawns_px()
    assert 24 * TILE in gs, gs[:5]
    print(f"OK: spawn={e}, chao tile3 fileira={c.floor_row(3)}, "
          f"pit tile30={c.floor_row(30)}, goombas px={gs[:6]}", flush=True)
    print("pit_ahead(48px,6):", c.pit_ahead(48.0), flush=True)
    print("pit_ahead(400px,6):", c.pit_ahead(400.0), flush=True)
