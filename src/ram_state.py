"""Direct game variable extraction via RAM memory reading (EUR ROM NTR-A2DP-EUR).

Discoveries validated via differential search + screenshots + memory pokes (see README §10):
  LIVES_ADDR  0x0209DC00 u8   : lives (5->4 on death; cf. AR code EUR 2209DC00)
  CAM_ADDRS   0x02098240 / 0x020DCFA0 u32 : camera X (deadzone, freezes on map)
  ODO_ADDR    0x0209AE9C u32  : horizontal activity counter
  MARIO_BASE  0x021C1890      : Mario object (type 0x1C at +0x44; stable
                                for this savestate across all tested boots)
  Object fields (validated offsets; Y=A, X=B in 20.12 fixed point):
    +0x44 u16 type (0x1C Mario, 0xA0 Goomba, ... table OBJ_TYPES)
    +0x60 s32 A (vertical; Mario jump arc, floor baseline -480px)
    +0x68 s32 B (horizontal; goomba marches left)
  Circular linked list of entities: node = obj+0x38 [prev, next];
  enemies link into list upon screen proximity (goomba links at ~20 steps).

Unresolved: absolute screen pixel X (two consistent bases: 0x021C1828 spawns
at 48.0px; object B differs by +40px — deltas are identical, which suffices
for reward); pixel Y is analogous. Enemies: RELATIVE position (dx, dy) to Mario.
"""
import numpy as np

LIVES_ADDR = 0x0209DC00
CAM_ADDRS = (0x02098240, 0x020DCFA0)
ODO_ADDR = 0x0209AE9C
Y_ADDR = 0x020A703C  # Experimental (flat on ground, increases in freefall)
VEL_ADDRS = (0x021C1904, 0x021C1928, 0x021C1940, 0x021C1A94, 0x021C1F6C)
MARIO_BASE = 0x021C1890
OFF_TYPE, OFF_A, OFF_B, OFF_NODE = 0x44, 0x60, 0x68, 0x38
SCAN_LO, SCAN_HI = 0x021C0000, 0x021E0000

OBJ_TYPES = {0x1C: "Mario", 0xA0: "Goomba", 0xA3: "Koopa", 0xA4: "Paratroopa",
             0x9E: "Coin", 0x9F: "StarCoin", 0x2C: "Powerup", 0xC8: "MovPlat",
             0x51: "HammerBro", 0x35: "Lakitu", 0x37: "Boo", 0x45: "DryBones",
             0x44: "Buzzy", 0x36: "Spiney", 0x1F: "Piranha", 0xE3: "ChompPlant",
             0xE8: "Pokey", 0x4C: "MegaDrop", 0x133: "TempProj"}

_U32_MOD = 2 ** 32


def _s32(v):
    return v - _U32_MOD if v >= 2 ** 31 else v


def _du32(cur, prev):
    """Delta calculation with wrap-around handling."""
    return (cur - prev + 2 ** 31) % _U32_MOD - 2 ** 31


class RamState:
    """RAM state reader. Cost per poll: ~10 memory reads + linked list walk (~50)."""

    def __init__(self, emu):
        self.emu = emu
        self.prev_cam = None
        self.acc_cam = 0.0
        self.max_acc = 0.0
        self.prev_lives = None
        self.deaths = 0
        self.mario_base = None

    # -- memory access primitives ---------------------------------------
    def _r32(self, addr):
        return self.emu.memory.read(addr, addr, 4, False)

    def _rs32(self, addr):
        return _s32(self._r32(addr))

    def _r16(self, addr):
        return self.emu.memory.read(addr, addr, 2, False)

    # -- Mario base discovery -------------------------------------------
    def _walk(self, start_node, maxn=40):
        seen, out, node = set(), [], start_node
        for _ in range(maxn):
            if node == 0 or node in seen or not (0x02000000 <= node < 0x02400000):
                break
            seen.add(node)
            nxt = self._r32(node + 4)
            obj = node - OFF_NODE
            out.append((obj, self._r16(obj + OFF_TYPE)))
            node = nxt
        return out

    def discover_mario(self):
        """Locates Mario's base address: known base (1 read) or scan + walk."""
        try:
            if self._r16(MARIO_BASE + OFF_TYPE) == 0x1C:
                self.mario_base = MARIO_BASE
                return self.mario_base
        except Exception:
            pass
        raw = bytes(self.emu.memory.read(SCAN_LO, SCAN_HI, 1, False))
        h = np.frombuffer(raw, dtype="<u2")
        for p in np.nonzero(h == 0x1C)[0][:400]:
            base = SCAN_LO + int(p) * 2 - OFF_TYPE
            if base < 0x02000000:
                continue
            try:
                nodes = self._walk(base + OFF_NODE)
            except Exception:
                continue
            if len(nodes) >= 2 and nodes[0][1] == 0x1C:
                self.mario_base = base
                return base
        self.mario_base = None
        return None

    def _mario_pos(self):
        if self.mario_base is None:
            return None, None
        try:
            return (self._rs32(self.mario_base + OFF_A),
                    self._rs32(self.mario_base + OFF_B))
        except Exception:
            return None, None

    def enemies(self, max_dx_px=600, max_dy_px=400):
        """Linked list walk starting from Mario: [{type, B, A, dx, dy}] (20.12 units).
        Filters entities by screen relevance (dx/dy in px). Validated Goomba:
        dx 164px -> contact (~10px) -> death via lives decrement."""
        if self.mario_base is None and self.discover_mario() is None:
            return []
        ma, mb = self._mario_pos()
        out = []
        try:
            nodes = self._walk(self.mario_base + OFF_NODE)
        except Exception:
            return []
        for obj, typ in nodes:
            if typ == 0x1C:
                continue
            try:
                a = self._rs32(obj + OFF_A)
                b = self._rs32(obj + OFF_B)
            except Exception:
                continue
            dx = (b - mb) / 4096.0 if mb is not None else None
            dy = (a - ma) / 4096.0 if ma is not None else None
            if dx is None or abs(dx) > max_dx_px or abs(dy or 0) > max_dy_px:
                continue
            e = {"type": typ, "name": OBJ_TYPES.get(typ, f"0x{typ:04X}"),
                 "A": a, "B": b, "dx": dx, "dy": dy}
            out.append(e)
        return out

    # -- main poll --------------------------------------------------------
    def poll(self):
        """Returns dict with current state + deltas. Called once per environment step."""
        mem = self.emu.memory
        lives = mem.read(LIVES_ADDR, LIVES_ADDR, 1, False)
        cam = mem.read(CAM_ADDRS[1], CAM_ADDRS[1], 4, False)
        odo = mem.read(ODO_ADDR, ODO_ADDR, 4, False)
        y = mem.read(Y_ADDR, Y_ADDR, 4, False)
        vel = _s32(mem.read(VEL_ADDRS[0], VEL_ADDRS[0], 4, False))

        if self.prev_cam is None:
            self.prev_cam, self.prev_lives = cam, lives
        cam_dx = _du32(cam, self.prev_cam)
        # Ignore anomalous jumps (area/map transitions): sanity deadzone
        if abs(cam_dx) > 1_000_000:
            cam_dx = 0
        self.acc_cam += cam_dx
        if self.acc_cam > self.max_acc:
            progress = self.acc_cam - self.max_acc
            self.max_acc = self.acc_cam
        else:
            progress = 0.0
        died = lives < self.prev_lives
        if died:
            self.deaths += 1
        self.prev_cam, self.prev_lives = cam, lives
        ma, mb = self._mario_pos()
        return {"lives": lives, "cam": cam, "cam_dx": cam_dx,
                "acc_cam": self.acc_cam, "progress": progress,
                "odo": odo, "y": y, "vel": vel, "died": died,
                "deaths": self.deaths, "mario_A": ma, "mario_B": mb}

    def reset(self):
        self.prev_cam = None
        self.acc_cam = 0.0
        self.max_acc = 0.0
        self.prev_lives = None
        # mario_base persists (same savestate => identical memory layout); rediscover on failure
