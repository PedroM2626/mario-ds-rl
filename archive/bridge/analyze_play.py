"""Analisa os 70 despejos como uma serie temporal e filtra enderecos.

Filtros (todos vetorializados, processados em blocos de colunas para caber em
memoria):
  velocidade  : |delta| <= 10 px/frame em QUALQUER par consecutivo
                (acima disso nao e posicao de personagem: e timer/animacao)
  patamar     : pelo menos 2 pares consecutivos sem mudanca (o jogador parado)
  variou      : mudou em algum momento da sessao
  faixa       : amplitude <= 8000 px e valores compativeis com uma fase
Extra: procura trios adjacentes (addr, +4, +8) que sobrevivem juntos, porque o
Actor do jogo guarda a posicao como vetor de 3 palavras de 32 bits.
"""
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 0x02000000
FIX = 65536.0
MAX_PX_PER_FRAME = 10.0
MAX_SPAN_PX = 8000.0

rows = list(csv.DictReader(open(os.path.join(HERE, "ramdir", "log.csv"))))
tags = [r["tag"] for r in rows]
frames = np.array([int(r["frame"]) for r in rows], dtype=np.int64)
gaps = np.diff(frames)
print(f"{len(tags)} despejos; frames {frames[0]}..{frames[-1]}; "
      f"gap min={gaps.min()} max={gaps.max()}")

WORDS = 4194304 // 4
stack = np.empty((len(tags), WORDS), dtype=np.int32)
for i, t in enumerate(tags):
    path = os.path.join(HERE, f"ram_{t}.bin")
    stack[i] = np.fromfile(path, dtype=np.int32, count=WORDS)
print("matriz carregada:", stack.shape, f"{stack.nbytes/1048576:.0f} MiB")

survivors = []
CHUNK = 32768
for c0 in range(0, WORDS, CHUNK):
    S = stack[:, c0:c0 + CHUNK].astype(np.int64)
    d = np.diff(S, axis=0)
    # velocidade por frame de cada transicao
    spd = np.abs(d) / (gaps[:, None] * FIX)
    speed_ok = (spd <= MAX_PX_PER_FRAME).all(axis=0)
    changed = (d != 0).any(axis=0)
    plateau = ((d == 0).sum(axis=0) >= 2)
    span = (S.max(axis=0) - S.min(axis=0)) / FIX
    range_ok = (S.min(axis=0) >= -4000 * FIX) & (S.max(axis=0) <= 90000 * FIX)
    ok = speed_ok & changed & plateau & (span <= MAX_SPAN_PX) & range_ok
    idx = np.nonzero(ok)[0]
    for k in idx:
        w = c0 + int(k)
        series = S[:, k] / FIX
        survivors.append({
            "addr": BASE + 4 * w,
            "span_px": float(span[k]),
            "moves": int((d[:, k] != 0).sum()),
            "plateaus": int((d[:, k] == 0).sum()),
            "first": float(series[0]), "last": float(series[-1]),
            "min": float(series.min()), "max": float(series.max()),
            "series": series.astype(np.float64).tolist(),
        })

print(f"\n{len(survivors)} enderecos passaram em todos os filtros")
survivors.sort(key=lambda s: -s["span_px"])
for s in survivors[:25]:
    print(f"  0x{s['addr']:08X}  faixa[{s['min']:9.2f}..{s['max']:9.2f}] "
          f"span={s['span_px']:8.2f}px  moveu={s['moves']:>2}/69  parou={s['plateaus']:>2}")

# trios adjacentes: assinatura de um vetor de posicao
addrs = {s["addr"]: s for s in survivors}
print("\n== trios adjacentes (addr, +4, +8) -- assinatura de vetor de posicao ==")
found = 0
for a, s in addrs.items():
    if (a + 4) in addrs and (a + 8) in addrs:
        found += 1
        print(f"  0x{a:08X}: X span={s['span_px']:.1f}px  "
              f"Y span={addrs[a+4]['span_px']:.1f}px  Z span={addrs[a+8]['span_px']:.1f}px")
        if found >= 8:
            break
if not found:
    print("  nenhum trio completo; procurando pares (addr, +4)...")
    for a, s in addrs.items():
        if (a + 4) in addrs:
            print(f"  par 0x{a:08X} span={s['span_px']:.1f} + 0x{a+4:08X} "
                  f"span={addrs[a+4]['span_px']:.1f}")
            found += 1
            if found >= 10:
                break

import json
json.dump([{k: v for k, v in s.items() if k != "series"} for s in survivors[:400]],
          open(os.path.join(HERE, "survivors.json"), "w"), indent=1)
print(f"\nsalvo em survivors.json ({min(len(survivors),400)} registros)")
