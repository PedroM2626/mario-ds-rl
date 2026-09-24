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
# Empirical kinematics grounded via 60 Hz DeSmuME telemetry (Route B decomposition):
# Running leap: 52 frames, 109.73 px reach, apex at frame 26 (71.06 px high, ~55 px forward)
# Walking jump: 52 frames, 75.33 px reach, apex at frame 26 (71.06 px high)
# Short hop: 22 frames, 19.94 px high
JUMP_REACH_PX = 109.73          # empirical running-jump horizontal reach
JUMP_REACH_TILES = int(JUMP_REACH_PX // TILE)  # 6 tiles (96 px safe margin)


def dynamic_jump_reach_px(vx_px_per_frame=None, airborne_frames=52.0):
    """Computes ballistic horizontal reach using empirical DeSmuME kinematics:
    X_reach = vx * t_airborne.
    Walking (1.47 px/f) -> 75.33 px (~4.7 tiles).
    Running/dash (2.14 px/f) -> 109.73 px (~6.8 tiles).
    Standing (0.0 px/f) -> 0 px (pure vertical leap: 63.81 px high).
    """
    vx = max(0.0, float(vx_px_per_frame if vx_px_per_frame is not None else 2.136))
    return vx * airborne_frames


def dynamic_jump_reach_tiles(vx_px_per_frame=None, airborne_frames=52.0):
    return int(dynamic_jump_reach_px(vx_px_per_frame, airborne_frames) // TILE)


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
        # walls/pipes = solid rects that rise well above the walking surface (a
        # running Mario collides and must *leap* them). Store the obstacle height per
        # column so short 1-2 tile steps are walked up, and only tall walls/pipes
        # (>= min_wall_tiles) trigger a leap -- leaping at small steps just throws
        # Mario into nearby enemies.
        self.obstacle_h = {}     # col -> tiles the wall rises above the floor
        for r in course.rects:
            top, bot = r["ty"], r["ty"] + r["h"]
            for cc in range(r["tx"], r["tx"] + r["w"]):
                fr = course.floor_row(cc)
                if fr is not None and top <= fr - 2 and bot >= fr - 1:
                    h = fr - top
                    if h > self.obstacle_h.get(cc, 0):
                        self.obstacle_h[cc] = h
        self.obstacle_cols = set(self.obstacle_h)
        # Walkable SURFACE height per column: the top of the solid run contiguous with
        # the ground (ground + blocks/pipes/stairs stacked on it), ignoring floating
        # ?-blocks/coins that Mario cannot stand on from the floor. Row 30 = floor;
        # a smaller row = a higher surface. This unifies pits (None), pipes and the
        # end staircase into a single height profile the controller can follow.
        solid_rows = {}          # col -> set(rows) of any solid tile
        for r in course.rects:
            for cc in range(r["tx"], r["tx"] + r["w"]):
                s = solid_rows.setdefault(cc, set())
                for yy in range(r["ty"], r["ty"] + r["h"]):
                    s.add(yy)
        self.surface = {}        # col -> top walkable row (or absent = pit)
        for cc in range(self.min_col, self.max_col + 1):
            fr = course.floor_row(cc)
            if fr is None:
                continue
            top = fr
            rows = solid_rows.get(cc, set())
            while (top - 1) in rows:      # climb contiguous solids above the floor
                top -= 1
            self.surface[cc] = top
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

    def enemy_threat_ahead(self, mario_abs_px, enemies=None, lookahead_px=80.0):
        """Scans real-time enemies for collision threats ahead, especially near takeoff edges.
        Returns:
            threat_type: None | 'dash_leap' | 'jump_stomp' | 'retreat_wait'
            info: dict with threat metrics
        """
        if not enemies:
            return None, {}
        for e in enemies:
            dx = e.get("dx")
            dy = e.get("dy", 0.0)
            if dx is None or dx <= 0.0 or dx > lookahead_px or abs(dy) > 48.0:
                continue
            evx = e.get("vx", 0.0)
            col = int((mario_abs_px + dx) / TILE)
            # Check if enemy is right at a takeoff boundary
            is_takeoff = (col in self.takeoffs) or ((col + 1) in self.takeoffs)
            if is_takeoff:
                # Goomba blocking pit edge!
                if dx < 48.0:
                    return "dash_leap", {"dx": dx, "dy": dy, "col": col, "hazard": "takeoff_blocker"}
                else:
                    return "retreat_wait", {"dx": dx, "dy": dy, "col": col, "hazard": "takeoff_blocker"}
            if dx < 38.0:
                return "jump_stomp", {"dx": dx, "dy": dy, "col": col, "hazard": "flat_enemy"}
        return None, {}

    def should_jump(self, mario_abs_px, vx_px_per_frame=None, enemies=None):
        """True when Mario is at a pit takeoff edge, approaching an enemy, or needs a running leap.
        Returns (jump: bool, upcoming_pit_width_tiles: int, reason: str).
        """
        # 1. Dynamic enemy threat check
        threat, info = self.enemy_threat_ahead(mario_abs_px, enemies)
        if threat in ("dash_leap", "jump_stomp"):
            return True, 0
        elif threat == "retreat_wait":
            return False, 0

        col = int(mario_abs_px / TILE)
        nxt = col + 1
        reach_tiles = dynamic_jump_reach_tiles(vx_px_per_frame)
        if nxt not in self.solid:                       # hole immediately ahead
            width = self._gap_width_at(nxt)
            return True, width
        # wide pit: anticipate takeoff by one tile so the long hop clears the gap
        tk = col + 1
        if tk in self.takeoffs and self.takeoffs[tk][3] >= 3:
            return True, self.takeoffs[tk][3]
        return False, 0

    def wall_ahead(self, mario_abs_px, look_tiles=3, min_wall_tiles=3):
        """Tiles until the next tall pipe/wall Mario would run into (0 = none).
        Only walls rising >= ``min_wall_tiles`` above the floor count, so 1-2 tile
        steps are walked up rather than leapt into nearby enemies. Used to trigger a
        *running* leap while Mario still has forward speed, since a standstill jump
        is too short to clear a ~4-tile pipe."""
        col = int(mario_abs_px / TILE)
        for d in range(1, look_tiles + 1):
            if self.obstacle_h.get(col + d, 0) >= min_wall_tiles:
                return d
        return 0

    def surface_ahead(self, mario_abs_px, look_tiles=4):
        """Height profile of the walkable surface over the next columns.
        Returns (cur_row, [(col, row_or_None), ...]) where a smaller row is higher and
        None is a pit. The controller compares these to decide walk / hop / leap."""
        col = int(mario_abs_px / TILE)
        cur = self.surface.get(col)
        prof = [(col + d, self.surface.get(col + d)) for d in range(1, look_tiles + 1)]
        return cur, prof

    def _gap_width_at(self, col):
        for (a, b) in self.pits:
            if a <= col <= b:
                return b - a + 1
        return 0

    def pipe_ahead(self, mario_abs_px, look_tiles=5, min_height_tiles=2):
        """Distance in pixels to the next pipe/wall rising >= min_height_tiles,
        or None if no pipe in lookahead.
        Used to ensure Mario launches before X = 358 px for the X = 384 px pipe."""
        col = int(mario_abs_px / TILE)
        cur = self.surface.get(col, 30)
        for d in range(1, look_tiles + 1):
            ahead_col = col + d
            h = self.obstacle_h.get(ahead_col, 0)
            s = self.surface.get(ahead_col)
            rise = (cur - s) if (cur is not None and s is not None) else 0
            if h >= min_height_tiles or rise >= min_height_tiles:
                return ahead_col * TILE - mario_abs_px
        return None

    def is_narrow_corridor(self, mario_abs_px):
        """True when Mario is approaching or within the narrow elevated corridor (1472-1552px, cols 92-97),
        where the Goomba patrols at X=1498px."""
        col = int(mario_abs_px / TILE)
        return 91 <= col <= 97

    def goal_px(self):
        return self.max_col * TILE


if __name__ == "__main__":
    p = TilemapAStar()
    print(f"goal_col={p.goal_col} (px {p.goal_px()})  feasible={p.feasible}")
    print("pits (start,end,width tiles):", p.pit_widths)
    print("takeoffs:", {k: v for k, v in sorted(p.takeoffs.items())})
    wide = [v for v in p.pit_widths if v[2] > p.reach]
    print(f"pits beyond {p.reach}-tile run-jump reach:", wide)
