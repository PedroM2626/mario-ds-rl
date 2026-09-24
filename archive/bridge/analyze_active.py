"""Segunda passada, com a expectativa certa para uma sessao de gameplay real.

O criterio anterior exigia patamares (transicoes sem mudanca), o que favorece
coisas estaticas e produziu falso positivo. Aqui o jogador esta ativo, entao a
coordenada real deve mudar em QUASE TODA transicao.

Filtros:
  move_frac : fração de transicoes que mudaram o valor  (>= 0.55)
  velocidade: |delta| <= 12 px/frame em qualquer transicao
  amplitude : span <= 4000 px (caminhada de ~3 min)
  suavidade : media |delta| / span <= 0.6  (serie anda, nao salta)
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE, FIX = 0x02000000, 65536.0

rows = list(csv.DictReader(open(os.path.join(HERE, "ramdir", "log.csv"))))
tags = [r["tag"] for r in rows]
frames = np.array([int(r["frame"]) for r in rows], dtype=np.int64)
gaps = np.diff(frames).astype(np.float64)
N = len(tags)
print(f"{N} despejos, {len(gaps)} transicoes, gap medio {gaps.mean():.0f} frames")

WORDS = 4194304 // 4
stack = np.empty((N, WORDS), dtype=np.int32)
for i, t in enumerate(tags):
    stack[i] = np.fromfile(os.path.join(HERE, f"ram_{t}.bin"), dtype=np.int32, count=WORDS)

res = []
CHUNK = 32768
for c0 in range(0, WORDS, CHUNK):
    S = stack[:, c0:c0 + CHUNK].astype(np.float64)
    d = np.diff(S, axis=0)
    spd = np.abs(d) / (gaps[:, None] * FIX)
    move_frac = (d != 0).mean(axis=0)
    ok_speed = (spd <= 12.0).all(axis=0)
    span = (S.max(axis=0) - S.min(axis=0)) / FIX
    ok_span = span <= 4000.0
    # media de |delta| por coluna, ignorando zeros
    nz = (d != 0)
    mean_abs = np.where(nz.sum(axis=0) > 0,
                        np.abs(np.where(nz, d, 0)).sum(axis=0) / np.maximum(1, nz.sum(axis=0)),
                        0) / FIX
    smooth = np.where(span > 0, mean_abs / np.maximum(span, 1e-9), 9.9)
    ok = (move_frac >= 0.55) & ok_speed & ok_span & (smooth <= 0.6) & (span >= 20.0)
    for k in np.nonzero(ok)[0]:
        w = c0 + int(k)
        col = S[:, k] / FIX
        res.append({"addr": BASE + 4 * w, "move_frac": float(move_frac[k]),
                    "span": float(span[k]), "smooth": float(smooth[k]),
                    "maxspd": float(spd[:, k].max()), "series": col.tolist()})

res.sort(key=lambda r: -r["move_frac"])
print(f"\n{len(res)} enderecos mudam quase sempre, devagar e sem saltar:")
for r in res[:20]:
    print(f"  0x{r['addr']:08X}  move={r['move_frac']:.2f}  span={r['span']:8.1f}px  "
          f"vel_max={r['maxspd']:5.2f}px/f  suave={r['smooth']:.2f}")

if res:
    print("\n== series dos 6 melhores ==")
    for r in res[:6]:
        s = " ".join(f"{v:.0f}" for v in r["series"])
        print(f"  0x{r['addr']:08X}: {s}")

# adjacencia: X e Y devem ser vizinhos (Vec3_32 position)
addrs = {r["addr"] for r in res}
print("\n== vizinhos adjacentes (assinatura de vetor de posicao) ==")
pairs = [a for a in addrs if (a + 4) in addrs]
for a in sorted(pairs)[:12]:
    ra = next(r for r in res if r["addr"] == a)
    rb = next(r for r in res if r["addr"] == a + 4)
    print(f"  0x{a:08X} (move={ra['move_frac']:.2f} span={ra['span']:7.1f})  "
          f"+4 -> 0x{a+4:08X} (move={rb['move_frac']:.2f} span={rb['span']:7.1f})")
print(f"  total de pares adjacentes: {len(pairs)}")
