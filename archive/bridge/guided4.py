"""Guia v4: amostra a 60 Hz uma lista ampliada, sem depender de relogio.

Voce le o roteiro ANTES e dispara com o seu "pronto", entao a unica coordenacao
que existe e a sua atencao. A lista de enderecos inclui os 38 candidatos do v2
mais os vizinhos imediatos, porque a posicao do Actor e um vetor de 3 palavras
de 32 bits (X, Y, Z contiguos) e o par correto pode nao ter passado no filtro.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

b = nsmb_env.Bridge()
st = b.status()
print("[status]", st, flush=True)
if st.get("emulating") != 1:
    raise SystemExit("ABORTADO: emulando != 1.")

base_hits = json.load(open(os.path.join(HERE, "guided2_hits.json")))
addrs = set()
for h in base_hits:
    a = h["addr"]
    for d in (-8, -4, 0, 4, 8, 12):
        addrs.add(a + d)
addrs = sorted(a for a in addrs if 0x02000000 <= a < 0x02400000)
print(f"[guia] {len(addrs)} enderecos (38 candidatos + vizinhos contiguos)", flush=True)

if os.path.exists(os.path.join(HERE, "ack")):
    os.remove(os.path.join(HERE, "ack"))
NF = 2400
with open(os.path.join(HERE, "cmd") + ".tmp", "w") as f:
    f.write("sample v4 %d %s" % (NF, " ".join(hex(a) for a in addrs)))
os.replace(os.path.join(HERE, "cmd") + ".tmp", os.path.join(HERE, "cmd"))
print("[guia] amostrando %d frames (~40s). Siga o roteiro que voce leu." % NF, flush=True)

ackf = os.path.join(HERE, "ack")
t0 = time.time()
while time.time() - t0 < 300:
    if os.path.exists(ackf) and "sample done" in open(ackf, errors="ignore").read():
        os.remove(ackf)
        break
    time.sleep(0.25)

path = os.path.join(HERE, "sample_v4.csv")
data = np.genfromtxt(path, delimiter=",", skip_header=1)
fr, cols = data[:, 0], data[:, 1:] / 65536.0
print(f"[guia] {len(fr)} frames, {cols.shape[1]} enderecos", flush=True)

# sem saber quando cada coisa aconteceu, procuramos a janela de 300 frames com o
# maior deslocamento unidirecional e continuo: isso so existe numa coordenada.
W = 300
best = []
for j, a in enumerate(addrs):
    v = cols[:, j]
    dv = np.diff(v)
    tot = np.convolve(dv, np.ones(W), mode="valid")
    if len(tot) == 0:
        continue
    k = int(np.argmax(np.abs(tot)))
    seg = dv[k:k + W]
    nz = seg[seg != 0]
    if len(nz) < W * 0.5:
        continue                       # precisa mudar em quase todo frame
    sign = 1.0 if tot[k] > 0 else -1.0
    frac_same = float((np.sign(nz) == sign).mean())
    if frac_same < 0.9:
        continue
    speed = float(np.mean(nz))
    if not (0.2 <= abs(speed) <= 12.0):
        continue
    best.append({"addr": int(a), "displacement_px": float(tot[k]),
                 "px_per_frame": speed, "frac_mesmo_sinal": frac_same,
                 "janela_inicio_frame": int(fr[k]), "min": float(v.min()),
                 "max": float(v.max())})

best.sort(key=lambda r: -abs(r["displacement_px"]))
print(f"\n{len(best)} enderecos com movimento unidirecional continuo e sustentado:")
for r in best[:15]:
    print(f"  0x{r['addr']:08X}  desloca={r['displacement_px']:+9.1f}px em {W} frames  "
          f"vel={r['px_per_frame']:+.3f}px/frame  coerencia={r['frac_mesmo_sinal']:.2f}  "
          f"faixa=[{r['min']:.1f}..{r['max']:.1f}]")

# pares adjacentes entre os melhores: X e Y vizinhos
tops = {r["addr"] for r in best[:40]}
print("\n== pares adjacentes entre os melhores (X e Y contiguos) ==")
for a in sorted(tops):
    if (a + 4) in tops:
        ra = next(r for r in best if r["addr"] == a)
        rb = next(r for r in best if r["addr"] == a + 4)
        print(f"  0x{a:08X} vel={ra['px_per_frame']:+.3f} desl={ra['displacement_px']:+8.1f}  |  "
              f"0x{a+4:08X} vel={rb['px_per_frame']:+.3f} desl={rb['displacement_px']:+8.1f}")

out = [{k: (bool(v) if isinstance(v, (np.bool_,)) else
            (float(v) if isinstance(v, (np.floating,)) else
             (int(v) if isinstance(v, (np.integer,)) else v)))
        for k, v in r.items()} for r in best[:60]]
json.dump(out, open(os.path.join(HERE, "guided4_hits.json"), "w"), indent=1)
print("\nsalvo em guided4_hits.json")
