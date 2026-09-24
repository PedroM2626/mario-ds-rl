"""Imprime as series de enderecos especificos para inspecao visual.

Serie em escada (patamares longos + rampas) = posicao de objeto controlado pelo
jogador. Serie dente-de-serra ou saltando sem padrao = timer/animacao/tabela.
"""
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE, FIX = 0x02000000, 65536.0
rows = list(csv.DictReader(open(os.path.join(HERE, "ramdir", "log.csv"))))
tags = [r["tag"] for r in rows]
frames = np.array([int(r["frame"]) for r in rows], dtype=np.int64)

WANT = [int(x, 16) for x in sys.argv[1:]] or [0x020A7678, 0x020A6C74, 0x020A6C3C,
                                              0x020A6C40, 0x020A6C44, 0x021C1F5C]

cache = {}
def word(addr, idx):
    t = tags[idx]
    if t not in cache:
        cache[t] = np.fromfile(os.path.join(HERE, f"ram_{t}.bin"), dtype=np.int32)
    off = (addr - BASE) // 4
    a = cache[t]
    return int(a[off]) if 0 <= off < len(a) else None

for addr in WANT:
    ser = np.array([word(addr, i) for i in range(len(tags))], dtype=np.float64) / FIX
    d = np.diff(ser)
    # "escada": fracao de transicoes pequenas e sinal consistente
    nz = d[d != 0]
    print(f"\n0x{addr:08X}  min={ser.min():9.2f} max={ser.max():9.2f} "
          f"unchanged={int((d==0).sum())}/{len(d)}  passos_uteis={len(nz)}")
    if len(nz):
        print(f"   |passo|: mediana={np.median(np.abs(nz)):8.2f}px  "
              f"max={np.abs(nz).max():8.2f}px   sinal: +{int((nz>0).sum())} / -{int((nz<0).sum())}")
    print("   serie:", " ".join(f"{v:.0f}" for v in ser[:35]))
    if len(ser) > 35:
        print("         ", " ".join(f"{v:.0f}" for v in ser[35:]))
