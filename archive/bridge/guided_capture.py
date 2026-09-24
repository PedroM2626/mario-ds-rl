"""Captura diferencial GUIADA: quem aperta as teclas e VOCE, nao o script.

Prova-se que SendInput nao chega ao DS, entao o input aqui e humano. As janelas
sao longas de proposito: o discriminador nao precisa de alinhamento exato, ele
so pede que o endereco mude na janela do meio e congele nas duas pontas.

Cronograma (depois dos 45s de tolerancia):
  [PARADO] 13s  -> despejo A
  [ANDA]   13s  -> despejo B   (segure a seta DIREITA o tempo todo)
  [PARA]    6s  -> despejo C   (solte tudo, sem apertar nada)
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env, ramdiff

b = nsmb_env.Bridge()
print("[status]", b.status(), flush=True)

GRACE = 45
print(f"[guia] {GRACE}s de tolerancia para ler as instrucoes...", flush=True)
time.sleep(GRACE)

print("[guia] FASE 1/3: FIQUE PARADO 13s (nao toque em nada)", flush=True)
time.sleep(13)
A = b.dump("g_a", 0x02000000, 0x400000)
print("[guia] despejo A gravado", flush=True)

print("[guia] FASE 2/3: SEGURE SETA DIREITA por 13s", flush=True)
time.sleep(13)
B = b.dump("g_b", 0x02000000, 0x400000)
print("[guia] despejo B gravado", flush=True)

print("[guia] FASE 3/3: SOLTE TUDO e fique parado 6s", flush=True)
time.sleep(6)
C = b.dump("g_c", 0x02000000, 0x400000)
print("[guia] despejo C gravado; analisando", flush=True)

moved = ramdiff.candidates_pair(A, B, direction="right")
print(f"[guia] {len(moved)} enderecos aumentaram durante a caminhada")

import struct
def words(blob):
    return struct.unpack_from(f"<{len(blob)//4}i", blob, 0)
wa, wb, wc = words(A), words(B), words(C)
hits = []
for i in range(len(wa)):
    d1 = wb[i] - wa[i]
    d2 = wc[i] - wb[i]
    if d1 == 0 or d2 != 0:
        continue
    if not (0 <= wc[i]/65536.0 <= 60000):
        continue
    if abs(d1)/65536.0 > 12.0 * 400:
        continue
    hits.append({"addr": ramdiff.BASE + 4*i,
                 "px_A": wa[i]/65536.0, "px_B": wb[i]/65536.0, "px_C": wc[i]/65536.0,
                 "delta_px": d1/65536.0})

hits.sort(key=lambda r: -abs(r["delta_px"]))
print(f"\n{len(hits)} enderecos mudaram enquanto voce andou e congelaram quando parou:")
for r in hits[:25]:
    print(f"  0x{r['addr']:08X}  A={r['px_A']:10.3f}  B={r['px_B']:10.3f}  C={r['px_C']:10.3f}"
          f"   delta_andou={r['delta_px']:+9.3f} px")

import json
json.dump(hits[:200], open(os.path.join(HERE, "guided_hits.json"), "w"), indent=1)
print("\nsalvo em guided_hits.json")
