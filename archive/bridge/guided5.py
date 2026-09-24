"""Protocolo final: sem coreografia. Voce JOGA NORMAL por ~60s e eu procuro.

Por que assim: as tentativas anteriores falharam por depender de timing humano
combinado por chat (que chega atrasado) e de uma lista de candidatos herdada de
uma coleta contaminada por pausa. Aqui nao ha o que sincronizar.

O discriminador e estrutural e nao pede saber quando voce andou: uma coordenada
de personagem e o UNICO tipo de dado em 4 MiB que sobe (ou desce) de forma
estrita por muitos despejos consecutivos enquanto voce anda, e congela bit a bit
quando voce para. Exigimos run monotono longo + platos nas pontas.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

NDUMPS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
GAP = 2.5
BASE = 0x02000000
WORDS = 4194304 // 4

b = nsmb_env.Bridge()
st = b.status()
print("[status]", st, flush=True)
if st.get("emulating") != 1:
    raise SystemExit("ABORTADO: emulando != 1 -- jogo pausado ou em menu.")

print(f"\n>>> JOGA NORMAL por ~{int(NDUMPS*GAP)}s: anda, corre, pula, pega moeda, para "
      f"de vez em quando. Nao precisa seguir ritmo nenhum. <<<\n", flush=True)

snaps, frames = [], []
for i in range(NDUMPS):
    t0 = time.time()
    snaps.append(np.frombuffer(b.dump(f"v5_{i}", BASE, 4194304), dtype=np.int32))
    f = int(b.status().get("framecount", 0))
    frames.append(f)
    print(f"  [{i+1:>2}/{NDUMPS}] frame={f}", flush=True)
    time.sleep(max(0.2, GAP - (time.time() - t0)))

S = np.vstack(snaps).astype(np.int64)
print(f"\nmatriz {S.shape}; frames {frames[0]}..{frames[-1]}", flush=True)
if frames[-1] - frames[0] < 60 * int(NDUMPS * GAP) * 0.4:
    print("AVISO: poucos frames avancaram durante a coleta; resultado pode ser lixo.")

d = np.diff(S, axis=0)
pos_run = np.zeros(S.shape[1], dtype=np.int32)
neg_run = np.zeros(S.shape[1], dtype=np.int32)
best_run = np.zeros(S.shape[1], dtype=np.int32)
best_dir = np.zeros(S.shape[1], dtype=np.int8)
for k in range(d.shape[0]):
    up = d[k] > 0
    dn = d[k] < 0
    pos_run = np.where(up, pos_run + 1, 0)
    neg_run = np.where(dn, neg_run + 1, 0)
    better = pos_run > best_run
    best_run = np.where(better, pos_run, best_run)
    best_dir = np.where(better, 1, best_dir)
    better = neg_run > best_run
    best_run = np.where(better, neg_run, best_run)
    best_dir = np.where(better, -1, best_dir)

# platô: ao menos 2 transicoes sem mudanca em algum ponto (voce parado)
plateau = (d == 0).sum(axis=0) >= 2
MINRUN = max(6, NDUMPS // 4)
cand = np.nonzero((best_run >= MINRUN) & plateau)[0]
print(f"{len(cand)} enderecos com run monotono >= {MINRUN} e ao menos um platô\n")

out = []
for w in cand:
    col = S[:, w] / 65536.0
    rng = float(col.max() - col.min())
    if not (5.0 <= rng <= 6000.0):
        continue
    out.append({"addr": BASE + 4*int(w), "run": int(best_run[w]),
                "dir": int(best_dir[w]), "range_px1616": rng,
                "series": [round(float(x), 3) for x in col]})
out.sort(key=lambda r: -r["run"])
for r in out[:15]:
    print(f"  0x{r['addr']:08X} run={r['run']} dir={r['dir']:+d} faixa={r['range_px1616']:8.2f}")
    print(f"     " + " ".join(f"{v:.0f}" for v in r["series"]))

json.dump(out[:200], open(os.path.join(HERE, "guided5_hits.json"), "w"), indent=1)
print(f"\nsalvo em guided5_hits.json ({len(out)} registros)")
