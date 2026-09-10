"""Env de GRAFO (Box(134) flat): no Mario + <=5 inimigos + <=8 retangulos
estaticos proximos (da ROM via course.py) + vetor global + mascara.

Recompensa/done/acoes identicos ao MarioRamEnv (heranca); so a observacao
muda (relacional em vez de vetor ordenado).
"""
import os
import sys
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ram_env import MarioRamEnv
from gnn_extractor import GLOBAL_DIM, N_NODES, NODE_F, OBS_DIM

N_EN = 5
N_ST = 8
TILE = 16


class MarioGraphEnv(MarioRamEnv):
    def __init__(self, rom_path=None, state_path=None, max_steps=1000):
        kw = {}
        if rom_path is not None:
            kw["rom_path"] = rom_path
        if state_path is not None:
            kw["state_path"] = state_path
        super().__init__(max_steps=max_steps, **kw)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(OBS_DIM,), dtype=np.float32)
        if self.course is None:
            try:
                from course import Course
                self.course = Course(rom_path=self.rom_path)
            except Exception as e:
                print(f"Sem course (nos estaticos zerados): {e}")

    def _observe(self):
        st = self.ram.poll()
        mb, ma = st["mario_B"], st["mario_A"]
        if self._x0 is None:
            self._x0, self._y0 = (mb or 0), (ma or 0)
        mx = ((mb or self._x0) - self._x0) / 4096.0
        my = ((ma or self._y0) - self._y0) / 4096.0
        vx = st["vel"] / 4096.0 / 4.0
        vy = self._s32read(0x021C1908) / 4096.0 / 4.0
        glob = [mx / 512.0, my / 512.0, vx, vy,
                1.0 if vy == 0.0 else 0.0,
                st["lives"] / 10.0,
                self._s32read(0x020DC968) / 4096.0 / 400.0,
                st["acc_cam"] / 4096.0 / 512.0]
        nodes, mask = [], []
        # no 0: Mario (dx=dy=0 por definicao)
        nodes.append([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        mask.append(1.0)
        # inimigos (ate 5, quaisquer — ordem irrelevante p/ GNN)
        try:
            ens = self.ram.enemies()[:50]
        except Exception:
            ens = []
        for e in ens[:N_EN]:
            nodes.append([e["dx"] / 256.0, e["dy"] / 256.0, 1.0, 1.0,
                          0.0, 1.0, 0.0, e["type"] / 256.0])
            mask.append(1.0)
        while len([m for m in mask]) < 1 + N_EN:
            nodes.append([0.0] * NODE_F)
            mask.append(0.0)
        # retangulos estaticos proximos (ate 8 por |dx|)
        stat = []
        if self.course is not None and mb is not None:
            mario_px = mb / 4096.0
            mario_py = -(ma / 4096.0) if ma is not None else 0.0
            for r in self.course.rects:
                cx = (r["tx"] + r["w"] / 2) * TILE - mario_px
                if abs(cx) < 300:
                    top = r["ty"] * TILE - mario_py
                    stat.append((abs(cx), [cx / 256.0, top / 256.0,
                                           r["w"] / 32.0, r["h"] / 8.0,
                                           0.0, 0.0, 1.0, r["obj"] / 256.0]))
        stat.sort(key=lambda t: t[0])
        for _, f in stat[:N_ST]:
            nodes.append(f)
            mask.append(1.0)
        while len(mask) < N_NODES:
            nodes.append([0.0] * NODE_F)
            mask.append(0.0)
        obs = glob + [v for nd in nodes[:N_NODES] for v in nd] + mask[:N_NODES]
        return np.array(obs[:OBS_DIM], dtype=np.float32), st
