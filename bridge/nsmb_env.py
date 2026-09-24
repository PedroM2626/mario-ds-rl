"""Python side of the DeSmuME <-> ML bridge.

The Lua side (desmume_bridge.lua) is the file-protocol server; this module is
the client plus a Gymnasium-shaped environment.

Typical use:
    from nsmb_env import NSMBEnv
    env = NSMBEnv()            # needs bridge/symbols.json (see ramdiff.py)
    obs = env.reset()
    obs, r, done, info = env.step("right_jump")
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "dumps")

DESMUME = r"C:/Users/pedro/Downloads/desmume/DeSmuME_0.9.13_x64.exe"
ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
CMD, ACK, TRACE = (os.path.join(HERE, n) for n in ("cmd", "ack", "trace.csv"))

# Buttons exposed to the agent. Each action is a set held for one step.
BUTTONS = ["left", "right", "down", "up", "A", "B", "X", "Y", "L", "R"]
ACT = {
    "noop": [],
    "left": ["Left"],
    "right": ["Right"],
    "jump": ["A"],
    "run": ["B"],
    "right_jump": ["Right", "A"],
    "left_jump": ["Left", "A"],
    "right_run": ["Right", "B"],
    "crouch": ["Down"],
}
ACTIONS = list(ACT.keys())


class BridgeError(RuntimeError):
    pass


class Bridge:
    """Send a command, wait for the ack. No sockets: the Lua side is the server."""

    def send(self, cmd: str, timeout: float = 120.0) -> str:
        if os.path.exists(ACK):
            os.remove(ACK)
        tmp = CMD + ".tmp"
        with open(tmp, "w") as f:
            f.write(cmd)
        os.replace(tmp, CMD)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(ACK):
                try:
                    out = open(ACK, errors="ignore").read().strip()
                except OSError:
                    time.sleep(0.05)
                    continue
                if out:
                    os.remove(ACK)
                    return out
            time.sleep(0.02)
        raise BridgeError(f"timeout waiting for ack of: {cmd!r}")

    def ping(self) -> bool:
        try:
            return self.send("ping", timeout=5).startswith("pong")
        except BridgeError:
            return False

    def status(self) -> dict:
        """Clock and pause state from the EMULATOR itself.

        The Lua loop's own counter is not the game clock: it keeps running while
        the game is paused, and reading RAM then yields perfectly flat series that
        look like "frozen address" but are only a pause menu. Two wrong
        conclusions in this project came from that, so every capture gates on
        this call.
        """
        txt = self.send("status", timeout=30)
        out = {}
        for tok in txt.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = int(v) if v.lstrip("-").isdigit() else v
        return out

    def wait_running(self, seconds: float = 1.5) -> bool:
        """True only if the emulator is actually advancing frames."""
        a = self.status()
        time.sleep(seconds)
        b = self.status()
        adv = b.get("framecount", -1) - a.get("framecount", -1)
        print(f"[status] emulating={b.get('emulating')} framecount={a.get('framecount')}"
              f"->{b.get('framecount')} (advanced {adv})")
        return adv > 10 and b.get("emulating") == 1

    def dump(self, tag: str, base: int = 0x02000000, length: int = 0x400000) -> bytes:
        """Read emulated RAM. The Lua writes the file; we consume and delete it.

        Deleting matters: a 3-minute capture at 18 Hz is ~3000 files, and leaving
        them in the bridge directory made every subsequent operation slow enough
        to time out the protocol.
        """
        msg = self.send(f"dump {tag} {hex(base)} {hex(length)}", timeout=600)
        if "ok" not in msg:
            raise BridgeError(msg)
        path = os.path.join(HERE, f"ram_{tag}.bin")
        with open(path, "rb") as f:
            blob = f.read()
        try:
            os.remove(path)
        except OSError:
            pass
        return blob

    def hold(self, buttons) -> str:
        # NOTE: this build's Lua has no input-writing API (input only has
        # get/popup/read/registerhotkey), and synthetic key events do not reach
        # the emulated DS. Labelled captures need human input; see archive/README.md.
        return self.send("hold " + " ".join(buttons), timeout=10)

    def launch(self):
        """Open DeSmuME with the ROM. The Lua script still has to be started once
        from the GUI (Tools > Lua Scripting); do NOT touch Restart/Stop afterwards,
        both kill the emulator process on this build."""
        return subprocess.Popen([DESMUME, ROM],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_symbols(path=os.path.join(HERE, "symbols.json")):
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


class NSMBEnv:
    """Gymnasium-shaped environment over observed RAM.

    reset() does not restart the game (that needs an emulator save state); it
    clears the step counter and reward accumulator.
    """

    def __init__(self, symbols=None, frames_per_step=1, bridge=None):
        self.bridge = bridge or Bridge()
        self.symbols = symbols if symbols is not None else load_symbols()
        self.frames_per_step = frames_per_step
        self.step_n = 0
        self.prev = None
        if not self.symbols:
            print("[env] no symbols.json: observations stay empty until ramdiff.py "
                  "locates addresses", file=sys.stderr)

    def read_state(self) -> dict:
        blob = self.bridge.dump(f"obs{self.step_n}")
        st = {}
        for name, loc in self.symbols.items():
            off = loc["addr"] - 0x02000000 + loc.get("sub", 0)
            if off + 8 > len(blob):
                continue
            raw = struct.unpack_from("<ii", blob, off)
            scaled = loc.get("scale") == "16.16"
            st[name] = raw[0] / 65536.0 if scaled else float(raw[0])
            if loc.get("pair"):
                st[loc["pair"]] = (raw[1] / 65536.0) if scaled else float(raw[1])
        return st

    def reset(self):
        self.step_n = 0
        self.prev = self.read_state()
        return self.prev

    def step(self, action):
        name = ACTIONS[action] if isinstance(action, int) else action
        self.bridge.hold(ACT.get(name, []))
        for _ in range(self.frames_per_step):
            self.bridge.ping()
        obs = self.read_state()
        r = self.reward(self.prev, obs)
        self.prev = obs
        self.step_n += 1
        done = bool(obs.get("dead")) or bool(obs.get("goal"))
        return obs, r, done, {"action": name}

    def reward(self, prev, obs):
        """Default reward: horizontal progress, penalised by death. Tunable."""
        if not prev or "pos_x" not in prev or "pos_x" not in obs:
            return 0.0
        return obs["pos_x"] - prev["pos_x"] - 100.0 * float(bool(obs.get("dead")))

    def gym_spec(self):
        """Optional Gymnasium spaces, if the package is installed."""
        try:
            import gymnasium as gym
        except ImportError:
            return None
        return gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-1e9, high=1e9,
                                  shape=(len(self.symbols) * 2 + 1,), dtype="float32"),
            "action": gym.spaces.Discrete(len(ACTIONS)),
        })


if __name__ == "__main__":
    b = Bridge()
    print("ping ->", b.ping())
    print("status ->", b.status())
