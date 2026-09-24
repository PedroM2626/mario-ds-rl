"""Confirmacao: amostra a trinca candidata a 60 Hz enquanto o jogador anda.

A 60 Hz a rampa de uma coordenada e inconfundivel, e a inclinacao em unidades/
frame resolve a ESCALA -- que e a假设 que eu vinha embutindo nos filtros e que
ja falseificou resultados antes.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

ADDRS = [0x0209DF80, 0x0209DF88, 0x0209DF98, 0x0209DFA0, 0x0209DFA4, 0x0209DFA8,
         0x0209DFAC, 0x0209DFB0, 0x0209DFB4, 0x0209DFB8, 0x0209DFC4, 0x0209DFC8,
         0x0209DFE0, 0x0209DFF0, 0x0209E000]
NF = 1500

b = nsmb_env.Bridge()
st = b.status()
print("[status]", st, flush=True)
if st.get("emulating") != 1:
    raise SystemExit("ABORTADO: jogo nao esta emulando (pausado ou em menu).")

if os.path.exists(os.path.join(HERE, "ack")):
    os.remove(os.path.join(HERE, "ack"))
with open(os.path.join(HERE, "cmd") + ".tmp", "w") as f:
    f.write("sample conf %d %s" % (NF, " ".join(hex(a) for a in ADDRS)))
os.replace(os.path.join(HERE, "cmd") + ".tmp", os.path.join(HERE, "cmd"))
print(f"[amostrando] {NF} frames (~25s). ANDE ~5s PARA A DIREITA e depois PARAR.", flush=True)

ackf = os.path.join(HERE, "ack")
t0 = time.time()
while time.time() - t0 < 180:
    if os.path.exists(ackf) and "sample done" in open(ackf, errors="ignore").read():
        os.remove(ackf)
        break
    time.sleep(0.25)
else:
    raise SystemExit("TIMEOUT: sample nao terminou")

data = np.genfromtxt(os.path.join(HERE, "sample_conf.csv"), delimiter=",", skip_header=1)
fr, cols = data[:, 0], data[:, 1:]
print(f"[ok] {len(fr)} frames ({fr[0]:.0f}..{fr[-1]:.0f}), {cols.shape[1]} enderecos\n")

W = 240
print(f"{'endereco':<12} {'distintos':>9} {'janela_andou':>12} {'desloc_bruto':>13} "
      f"{'unidade/frame':>13} {'passo_medio':>11}")
for j, a in enumerate(ADDRS):
    v = cols[:, j]
    d = np.diff(v)
    tot = np.convolve(d, np.ones(W), mode="valid")
    if len(tot) == 0:
        continue
    k = int(np.argmax(np.abs(tot)))
    seg = d[k:k + W]
    nz = seg[seg != 0]
    disp = float(tot[k])
    per = disp / W
    print(f"0x{a:08X} {len(np.unique(v)):>9} {int(fr[k]):>12} {disp:>13.0f} "
          f"{per:>13.2f} {(float(np.mean(np.abs(nz))) if len(nz) else 0):>11.1f}"
          + ("   <== CANDIDATO A COORDENADA" if abs(per) > 0.5 and len(nz) > W * 0.5 else ""))

print("\n== serie do X candidato (0x0209DFA4), a cada 10 frames ==")
j = ADDRS.index(0x0209DFA4)
v = cols[:, j]
print("  " + " ".join(f"{x:.0f}" for x in v[::10]))
d = np.diff(v)
moving = np.abs(d) > 0
print(f"\nframes com mudanca: {moving.mean():.2%}   passo mediano quando muda: "
      f"{np.median(np.abs(d[d != 0])):.0f} unidades/frame")
print(f"min={v.min():.0f} max={v.max():.0f} amplitude={v.max()-v.min():.0f}")
