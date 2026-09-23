"""Tilemap A* jump-reachability planner for NSMB-DS levels (Task 3).

Parses level geometry from the ROM (`course.py`) and plans a forward path as a
sequence of *run-jump* hops between solid ground platforms, so the reactive/MPC
controller takes off from the exact edge column for each pit (rather than a crude
fixed pixel lookahead) and never stalls on ledges/walls.

Physics grounding
-----------------
With the dash action enabled, Mario reaches ~3 px/frame and a running jump spans
roughly `JUMP_REACH_PX` horizontally. A pit narrower than that can be cleared from
its near edge; wider pits need the plan to confirm feasibility (they would require
a higher/longer jump than the engine's walking-running jump allows).

The planner is deliberately console-side: it is the *global* layer of a hierarchical
controller (A* picks where to land and when to leap; CEM-MPC + reflex handle the
dynamic enemies and fine timing in the local horizon).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from course import Course, TILE  # noqa: E402

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
JUMP_REACH_PX = 120.0          # conservative running-jump horizontal reach
JUMP_REACH_TILES = int(JUMP_REACH_PX // TILE)  # ~7 tiles


class TilemapAStar:
    def __init__(self, rom_path=ROM, name="course/A01_1.bin", jump_reach_tiles=JUMP_REACH_TILES):
        course = Course(name=name, rom_path=rom_path)
        cols = sorted(course.ground_cover.keys())
        self.min_col, self.max_col = cols[0], cols[-1]
        self.solid = set(cols)
        self.goal_col = self.max_col
        self.reach = jump_reach_tiles
        # pits = maximal runs of missing columns between [min_col, max_col]
        self.pits = []          # (start_col, end_col) inclusive, missing ground
        c = self.min_col
        while c <= self.max_col:
            if c not in self.solid:
                a = c
                while c <= self.max_col and c not in self.solid:
                    c += 1
                self.pits.append((a, c - 1))
            c += 1
        self.pit_widths = [(a, b, b - a + 1) for (a, b) in self.pits]
        # takeoff column = last solid before a pit; landing = first solid after it
        self.takeoffs = {}      # takeoff_col -> (pit_start, pit_end, landing_col, width)
        for (a, b) in self.pits:
            tk = a - 1
            if tk in self.solid:
                self.takeoffs[tk] = (a, b, b + 1, b - a + 1)
        self.feasible = all((b - a + 1) <= self.reach for (a, b) in self.pits)

    def next_takeoff(self, mario_col):
        """Nearest required takeoff column strictly ahead of (or at) Mario. Returns
        dict with the takeoff col, pit width, and landing col, or None near the goal."""
        best = None
        for tk, (a, b, land, w) in self.takeoffs.items():
            if tk >= mario_col - 1:            # takeoff at/after current position
                if best is None or tk < best["takeoff_col"]:
                    best = {"takeoff_col": tk, "pit_start": a, "pit_end": b,
                            "landing_col": land, "width": w}
        return best

    def should_jump(self, mario_abs_px):
        """True when Mario is at a pit takeoff edge (next column is a hole) or, for
        a wide pit, within one tile of its takeoff -- the A*-prescribed leap.

        Returns (jump: bool, upcoming_pit_width_tiles: int).
        """
        col = int(mario_abs_px / TILE)
        nxt = col + 1
        if nxt not in self.solid:                       # hole immediately ahead
            width = self._gap_width_at(nxt)
            return True, width
        # wide pit: anticipate takeoff by one tile so the long hop clears the gap
        tk = col + 1
        if tk in self.takeoffs and self.takeoffs[tk][3] >= 3:
            return True, self.takeoffs[tk][3]
        return False, 0

    def _gap_width_at(self, col):
        for (a, b) in self.pits:
            if a <= col <= b:
                return b - a + 1
        return 0

    def goal_px(self):
        return self.max_col * TILE


if __name__ == "__main__":
    p = TilemapAStar()
    print(f"goal_col={p.goal_col} (px {p.goal_px()})  feasible={p.feasible}")
    print("pits (start,end,width tiles):", p.pit_widths)
    print("takeoffs:", {k: v for k, v in sorted(p.takeoffs.items())})
    wide = [v for v in p.pit_widths if v[2] > p.reach]
    print(f"pits beyond {p.reach}-tile run-jump reach:", wide)
