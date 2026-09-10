"""Busca diferencial de enderecos RAM na ROM EUR (onde os mapas US nao valem).

Fase 1: segura DIREITA, coleta N dumps; acha u32/s32/u16 monotonicamente
        crescentes (candidato a X do Mario / camera / contadores).
Uso:
  python src/ram_search.py --dumps 6 --steps 400 --out ram_search.npz
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
RAM_BASE, RAM_SIZE = 0x02000000, 4 * 1024 * 1024


def dump_ram(emu):
    return np.frombuffer(
        bytes(emu.memory.read(RAM_BASE, RAM_BASE + RAM_SIZE, 1, False)),
        dtype=np.uint8).copy()


def hold(emu, keys, steps, frameskip=8):
    for k in keys:
        emu.input.keypad_add_key(keymask(k))
    for _ in range(steps):
        for _ in range(frameskip):
            emu.cycle()
    for k in keys:
        emu.input.keypad_rm_key(keymask(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", type=str, default="ram_search.npz")
    args = ap.parse_args()

    emu = DeSmuME()
    emu.open(ROM)
    emu.savestate.load_file(STATE)

    dumps = [dump_ram(emu)]
    for i in range(args.dumps - 1):
        hold(emu, [Keys.KEY_RIGHT], args.steps)
        dumps.append(dump_ram(emu))
        print(f"dump {i+2}/{args.dumps} ok", flush=True)
    emu.destroy()

    D = np.stack(dumps)  # (N, 4MB) uint8
    print("procurando u32 monotonicos...", flush=True)
    U = np.stack([np.frombuffer(d.tobytes(), dtype="<u4") for d in dumps])
    S = U.view(np.int32)
    mono_u = np.ones(U.shape[1], bool)
    mono_s = np.ones(U.shape[1], bool)
    for k in range(1, len(dumps)):
        mono_u &= U[k] > U[k - 1]
        mono_s &= S[k] > S[k - 1]
    both = mono_u | mono_s
    idx = np.nonzero(both)[0]
    print(f"candidatos u32/s32 monotonicos: {len(idx)}", flush=True)
    for i in idx[:50]:
        vals = U[:, i].tolist()
        print(f"  0x{RAM_BASE + int(i)*4:08X} u32={vals} signed={S[:, i].tolist()}", flush=True)

    H = np.stack([np.frombuffer(d.tobytes(), dtype="<u2") for d in dumps])
    mono_h = np.ones(H.shape[1], bool)
    for k in range(1, len(dumps)):
        mono_h &= H[k] > H[k - 1]
    hidx = np.nonzero(mono_h)[0]
    print(f"candidatos u16 monotonicos: {len(hidx)} (top 20 por delta)", flush=True)
    deltas = (H[-1, hidx].astype(int) - H[0, hidx].astype(int))
    order = np.argsort(-deltas)[:20]
    for j in order:
        i = hidx[j]
        print(f"  0x{RAM_BASE + int(i)*2:08X} u16={H[:, i].tolist()}", flush=True)

    np.savez_compressed(args.out, dumps=D, u32_idx=idx, u16_idx=hidx)
    print(f"salvo em {args.out}", flush=True)


if __name__ == "__main__":
    main()
