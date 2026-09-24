"""Varre a vizinhanca do candidato sem precisar de nova coleta.

Os 25 despejos de ram_v5_*.bin continuam em disco. Aqui imprimimos a serie de
CADA palavra de 32 bits na janela ao redor de 0x0209DFA4, porque o Actor declara
os campos em ordem fixa (position, ..., velH, minVelH, accelV, minVelV,
accelH em 0xC4). Achar a BASE faz de accelV (gravidade) um offset fixo: 0xBC.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_RAM = 0x02000000
FIX = 65536.0

tags = [f"v5_{i}" for i in range(25)]
missing = [t for t in tags if not os.path.exists(os.path.join(HERE, f"ram_{t}.bin"))]
if missing:
    raise SystemExit(f"faltam despejos: {missing[:3]} ...")

CENTER = int(sys.argv[1], 16) if len(sys.argv) > 1 else 0x0209DFA4
SPAN = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0x90
lo = CENTER - SPAN
hi = CENTER + SPAN

words = (hi - lo) // 4
series = np.zeros((25, words), dtype=np.int64)
for i, t in enumerate(tags):
    with open(os.path.join(HERE, f"ram_{t}.bin"), "rb") as f:
        f.seek(lo - BASE_RAM)
        buf = f.read(words * 4)
    series[i] = np.frombuffer(buf, dtype=np.int32).astype(np.int64)

print(f"janela 0x{lo:08X}..0x{hi:08X}  ({words} palavras) x 25 despejos\n")
print(f"{'endereco':<12} {'distintos':>9} {'mudou':>6} {'faixa':>10}  serie (valores brutos /65536)")

for j in range(words):
    v = series[:, j]
    d = np.diff(v)
    rng = int(v.max() - v.min())
    if len(np.unique(v)) < 2:
        continue                      # constante nao interessa
    frac = float((d != 0).mean())
    line = " ".join(f"{x/FIX:.0f}" for x in v)
    print(f"0x{lo+4*j:08X} {len(np.unique(v)):>9} {frac:>6.2f} {rng:>10}  {line}")

print("\nReferencia: se a base do Actor for B, gravidade = B+0xBC, "
      "terminal = B+0xC0, accelH = B+0xC4, velH = B+0xB8.")
