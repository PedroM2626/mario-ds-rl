"""Diagnostico do sample_v4.csv SEM suposicao de escala.

Os filtros anteriores pressupoem ponto fixo 16.16 e velocidade em px/frame.
Se a escala for outra, isso descarta a coordenada correta. Aqui cada endereco e
descrito por propriedades adimensionais: quantos valores distintos, que
fracao dos frames mudaram, e qual o maior trecho mudando sempre no mesmo
sentido. Uma coordenada de jogador se destaca por coerencia direcional, seja
qual for a unidade.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "sample_v4.csv")
if not os.path.exists(path):
    raise SystemExit("sample_v4.csv nao existe")

head = open(path).readline().strip().split(",")
data = np.genfromtxt(path, delimiter=",", skip_header=1)
fr, cols = data[:, 0], data[:, 1:]
print(f"{len(fr)} frames x {cols.shape[1]} enderecos")
print(f"frames: {fr[0]:.0f}..{fr[-1]:.0f}  (delta={fr[-1]-fr[0]:.0f})")

rows = []
for j in range(cols.shape[1]):
    v = cols[:, j]
    d = np.diff(v)
    changed = d != 0
    nz = d[changed]
    if len(nz) == 0:
        continue
    # maior corrida consecutiva com mudanca no MESMO sentido
    sign = np.sign(nz)
    best_run = cur = 1
    for a, b in zip(sign, sign[1:]):
        cur = cur + 1 if (a == b and a != 0) else 1
        best_run = max(best_run, cur)
    rows.append({
        "addr": head[1 + j],
        "distinct": len(np.unique(v)),
        "frac_frames_changed": float(changed.mean()),
        "best_same_sign_run": int(best_run),
        "total_change": float(v[-1] - v[0]),
        "range": float(v.max() - v.min()),
        "median_step": float(np.median(np.abs(nz))),
    })

df = sorted(rows, key=lambda r: -(r["best_same_sign_run"] * r["frac_frames_changed"]))
print(f"\n{len(df)} enderecos mudaram em algum momento. Top por coerencia direcional:\n")
print(f"{'endereco':<12} {'distintos':>9} {'frac_mudou':>10} {'maior_corrida':>13} "
      f"{'faixa_bruta':>14} {'passo_medio':>11} {'variacao_total':>14}")
for r in df[:22]:
    print(f"{r['addr']:<12} {r['distinct']:>9} {r['frac_frames_changed']:>10.2f} "
          f"{r['best_same_sign_run']:>13} {r['range']:>14.0f} {r['median_step']:>11.0f} "
          f"{r['total_change']:>14.0f}")
