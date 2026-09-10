"""Env 100% RAM (sem pixels, sem CNN): estado = vetor de 17 floats.

Observacao (Box 17, normalizada p/ MLP):
  [x, y, vx, vy, on_ground, lives, time, cam_x,
   edx0, edy0, etype0, edx1, edy1, etype1, edx2, edy2, etype2]
  x,y relativos ao reset (px, /512); vx,vy em px/frame (/4);
  inimigos: 3 mais proximos em |dx|, dx/dy em px (/256), tipo/256.
Recompensa: mesma formula do env visual (progresso*2 - 0.05, morte -15).
Done: morte (vidas) ou timer zerado (-50, como timeout) ou 1000 steps.
Sem pixels: finish (+100) nao detectado nesta v1 (nenhum agente chega la em
testes curtos); timeout usa o timer-RAM.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask
except ImportError:
    DeSmuME = None

from ram_state import RamState, _s32

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
TIME_ADDR = 0x020DC968
YVEL_ADDR = 0x021C1908
N_ENEMIES = 3
OBS_DIM = 8 + 3 * N_ENEMIES
GEO_DIM = 6  # flags de pit nas 6 colunas a frente
MAX_STEPS = 1000


class MarioRamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rom_path=ROM, state_path=STATE, max_steps=MAX_STEPS, geo=False):
        super().__init__()
        self.rom_path = rom_path
        self.state_path = state_path
        self.max_steps = max_steps
        self.geo = geo
        self.action_space = spaces.Discrete(6)
        dim = OBS_DIM + (GEO_DIM if geo else 0)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(dim,), dtype=np.float32)
        self.course = None
        if geo:
            try:
                from course import Course
                self.course = Course(rom_path=rom_path)
            except Exception as e:
                print(f"Geo desativado (falha ao ler course: {e})")
                self.geo = False
        try:
            self.emu = DeSmuME()
            self.emu.open(self.rom_path)
            self.emu.savestate.load_file(self.state_path)
            self.has_emulator = True
        except Exception as e:
            print(f"Failed to initialize emulator: {e}")
            self.has_emulator = False
            self.emu = None
        self.ram = RamState(self.emu) if self.has_emulator else None
        self.frameskip = 8
        self._x0 = self._y0 = None

    # -- helpers ------------------------------------------------------
    def _s32read(self, addr):
        return _s32(self.emu.memory.read(addr, addr, 4, False))

    def _observe(self):
        st = self.ram.poll()
        mb, ma = st["mario_B"], st["mario_A"]
        if self._x0 is None:
            self._x0, self._y0 = (mb or 0), (ma or 0)
        x = ((mb or self._x0) - self._x0) / 4096.0 / 512.0
        y = ((ma or self._y0) - self._y0) / 4096.0 / 512.0
        vx = st["vel"] / 4096.0 / 4.0
        vy = self._s32read(YVEL_ADDR) / 4096.0 / 4.0
        on_ground = 1.0 if vy == 0.0 else 0.0
        lives = st["lives"] / 10.0
        time_raw = self._s32read(TIME_ADDR) / 4096.0
        self._last_time = time_raw
        time_left = time_raw / 400.0
        cam = st["acc_cam"] / 4096.0 / 512.0
        obs = [x, y, vx, vy, on_ground, lives, time_left, cam]
        ens = sorted(self.ram.enemies(), key=lambda e: abs(e["dx"] or 1e9))[:N_ENEMIES]
        for e in ens:
            obs += [e["dx"] / 256.0, e["dy"] / 256.0, e["type"] / 256.0]
        while len(obs) < OBS_DIM:
            obs += [0.0, 0.0, 0.0]
        obs = obs[:OBS_DIM]
        if self.geo and self.course is not None:
            mb = st["mario_B"]
            if mb is None:
                if self.ram.mario_base is None:
                    self.ram.discover_mario()
                mb, _ = self.ram._mario_pos()
            mario_px = (mb / 4096.0) if mb is not None else 0.0
            obs += self.course.pit_ahead(mario_px, GEO_DIM)
        return np.array(obs, dtype=np.float32), st

    # -- gym api ------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.has_emulator:
            self.emu.savestate.load_file(self.state_path)
            self.ram.reset()
        self._x0 = self._y0 = None
        self.episode_steps = 0
        obs, _ = self._observe()
        # fixa origem apos primeira leitura
        return obs, {}

    def step(self, action):
        keys = []
        if action == 1:
            keys.append(Keys.KEY_RIGHT)
        elif action == 2:
            keys.extend([Keys.KEY_RIGHT, Keys.KEY_B])
        elif action == 3:
            keys.extend([Keys.KEY_RIGHT, Keys.KEY_B, Keys.KEY_A])
        elif action == 4:
            keys.append(Keys.KEY_LEFT)
        elif action == 5:
            keys.append(Keys.KEY_A)
        for key in keys:
            self.emu.input.keypad_add_key(keymask(key))
        for _ in range(self.frameskip):
            self.emu.cycle()
        for key in keys:
            self.emu.input.keypad_rm_key(keymask(key))
        obs, st = self._observe()
        reward = st["progress"] / 4096.0 * 2.0 - 0.05
        done, trunc = False, False
        if st["died"]:
            done = True
            reward -= 15.0
        elif getattr(self, "_last_time", 1.0) <= 0:
            done = True
            reward -= 50.0
        self.episode_steps += 1
        if self.episode_steps >= self.max_steps:
            trunc = True
        return obs, reward, done, trunc, {"x": float(obs[0])}

    def close(self):
        if self.has_emulator:
            try:
                self.emu.destroy()
            except Exception:
                pass
