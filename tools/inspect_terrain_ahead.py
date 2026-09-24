import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.course import Course, TILE
from src.tilemap_planner import TilemapAStar

c = Course(rom_path='data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds')
p = TilemapAStar()

print("=== TERRAIN BREAKDOWN (Col 95 to 266) ===")
for col in range(95, 267):
    fr = c.floor_row(col)
    surf = p.surface.get(col)
    ob_h = p.obstacle_h.get(col, 0)
    is_solid = col in p.solid
    sprites = [s['name'] for s in c.sprites if s['tx'] == col]
    sp_str = f" | Sprites: {', '.join(sprites)}" if sprites else ""
    if not is_solid or ob_h > 0 or (surf is not None and surf < 30) or sp_str:
        print(f"Col {col:3d} (x={col*TILE:4d}px): solid={is_solid} floor={fr} surf={surf} ob_h={ob_h}{sp_str}")
