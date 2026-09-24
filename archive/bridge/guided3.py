"""Guia v3: amostra os 38 candidatos a 60 Hz enquanto voce anda e pula.

Com 38 enderecos conhecidos, o amostrador por frame resolve o problema de
timing: nao importa em que segundo voce andou, porque a rampa aparece nos dados
e a inclricao dela e a velocidade. Pulos aparecem como parabolas no eixo Y.

Cronometro: 60s de tolerancia, depois ~26s de amostragem continua.
  primeiros 5s : parado
  proximos 8s  : segure -> (direita) sem parar
  3s           : solte tudo, parado
  resto        : 4 pulos longos (segure A uns 0,5s), com 1s no chao entre eles
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
    raise SystemExit("ABORTADO: emulando != 1 (jogo pausado ou em menu).")

hits = json.load(open(os.path.join(HERE, "guided2_hits.json")))
addrs = [h["addr"] for h in hits]
print(f"[guia] {len(addrs)} candidatos do v2; 60s de tolerancia para ler o roteiro.", flush=True)
time.sleep(60)

NF = 1600
if os.path.exists(os.path.join(HERE, "ack")):
    os.remove(os.path.join(HERE, "ack"))
with open(os.path.join(HERE, "cmd") + ".tmp", "w") as f:
    f.write("sample v3 %d %s" % (NF, " ".join(hex(a) for a in addrs)))
os.replace(os.path.join(HERE, "cmd") + ".tmp", os.path.join(HERE, "cmd"))
print("[guia] AMOSTRANDO. AGORA: fique parado 5s", flush=True)
time.sleep(5)
print("[guia] >>> SEGURE -> (direita) por 8s <<<", flush=True)
time.sleep(8)
print("[guia] >>> SOLTE. parado 3s <<<", flush=True)
time.sleep(3)
print("[guia] >>> 4 PULOS LONGOS: segure A ~0,5s, 1s no chao entre eles <<<", flush=True)

ackf = os.path.join(HERE, "ack")
t0 = time.time()
while time.time() - t0 < 60:
    if os.path.exists(ackf) and "sample done" in open(ackf, errors="ignore").read():
        os.remove(ackf)
        break
    time.sleep(0.25)

path = os.path.join(HERE, "sample_v3.csv")
head = open(path).readline().strip().split(",")
data = np.genfromtxt(path, delimiter=",", skip_header=1)
fr = data[:, 0]
cols = data[:, 1:] / 65536.0
print(f"\n[guia] {len(fr)} frames amostrados ({fr[0]:.0f}..{fr[-1]:.0f}), "
      f"{cols.shape[1]} enderecos", flush=True)

print("\n== classificacao: rampa continua durante a caminhada (frames 300..780) ==")
win = slice(300, 780)
score = []
for j, a in enumerate(addrs):
    v = cols[win, j]
    d = np.diff(v)
    nz = d[d != 0]
    if len(nz) == 0:
        continue
    total = v[-1] - v[0]
    # rampa: muitos passos nao-zero, todos com o mesmo sinal, tamanho uniforme
    same_sign = max((nz > 0).sum(), (nz < 0).sum()) / len(nz)
    uniform = 1.0 / (1.0 + float(np.std(nz) / (abs(np.mean(nz)) + 1e-9)))
    step_frac = len(nz) / len(d)
    score.append({"addr": a, "total_px": float(total), "step_frac": float(step_frac),
                  "same_sign": float(same_sign), "uniform": float(uniform),
                  "px_per_frame": float(np.mean(nz)) if len(nz) else 0.0,
                  "rank": step_frac * same_sign * abs(total)})

score.sort(key=lambda r: -r["rank"])
for r in score[:12]:
    print(f"  0x{r['addr']:08X}  anda={r['total_px']:+8.2f}px  passos={r['step_frac']:.2f} "
          f"mesmo_sinal={r['same_sign']:.2f} vel_media={r['px_per_frame']:+.4f} px/frame")

print("\n== candidatos a Y: variam muito e voltam ao mesmo valor (pulos) ==")
y = []
for j, a in enumerate(addrs):
    v = cols[780:, j]
    if len(v) < 10:
        continue
    rng = float(v.max() - v.min())
    back = abs(v[-1] - v[0]) < 1.0
    if rng > 20.0 and back:
        y.append({"addr": a, "range_px": rng, "returns": back})
y.sort(key=lambda r: -r["range_px"])
for r in y[:10]:
    print(f"  0x{r['addr']:08X}  amplitude_no_ar={r['range_px']:8.2f}px  volta_ao_mesmo={r['returns']}")

json.dump({"walk": score[:20], "jump": y[:20]},
          open(os.path.join(HERE, "guided3_hits.json"), "w"), indent=1)
print("\nsalvo em guided3_hits.json")
