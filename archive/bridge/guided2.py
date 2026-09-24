"""Guia v2: seis despejos, exigindo CONTINUIDADE, nao so diferenca de pontas.

O v1 comparava A->B e estava errado: qualquer memoria escrita uma unica vez
passa. Endereco de posicao tem que mudar em TODA janela de caminhada e em NENHUMA
janela de repouso.

Cronograma (60s de tolerancia para ler):
  PARADO   8s -> despejo 1
  PARADO   8s -> despejo 2     (1 e 2 devem ser identicos)
  ANDA ->   8s -> despejo 3
  ANDA ->   8s -> despejo 4     (3 e 4 DEVEM diferir, no mesmo sentido)
  PARADO   8s -> despejo 5
  PARADO   8s -> despejo 6      (5 e 6 devem ser identicos)
"""
import os, sys, time, json, struct
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env, ramdiff

b = nsmb_env.Bridge()
print("[status]", b.status(), flush=True)
print("[guia] 60s de tolerancia. Depois: 2x parado, 2x ANDANDO ->, 2x parado.", flush=True)
time.sleep(60)

NAMES = ["1 PARADO", "2 PARADO", "3 ANDA", "4 ANDA", "5 PARADO", "6 PARADO"]
# Sem keybd_event aqui: ja provamos que SendInput nao chega ao DS neste setup.
# O input das fases "ANDA" e humano, guiado pelo cronometro abaixo.
blobs, frames = [], []
for i, nm in enumerate(NAMES):
    print(f"[guia] fase {nm}: 8s" + ("  >>> SEGURE SETA DIREITA <<<" if "ANDA" in nm else ""),
          flush=True)
    time.sleep(8)
    blobs.append(b.dump(f"v2_{i}", 0x02000000, 0x400000))
    frames.append(int(b.status().get("framecount", 0)))
    print(f"[guia] despejo {i+1} ok (frame {frames[-1]})", flush=True)

W = [struct.unpack_from(f"<{len(bl)//4}i", bl, 0) for bl in blobs]
N = min(len(x) for x in W)
gap = [frames[i+1] - frames[i] for i in range(5)]
print(f"\ngaps entre despejos: {gap}")

hits = []
for i in range(N):
    v = [W[j][i] for j in range(6)]
    if v[0] != v[1] or v[4] != v[5]:
        continue                        # tem que estar congelado nos repousos
    d34 = v[3] - v[2]
    if d34 == 0:
        continue                        # tem que mudar DURANTE a caminhada
    if v[2] == v[1] or v[3] == v[4]:
        pass                            # transicao esperada, nao obrigatoria
    px = d34 / 65536.0
    per_frame = px / max(1, gap[2])
    if not (0.2 <= abs(per_frame) <= 12.0):
        continue
    if not (-2000.0 <= v[5]/65536.0 <= 60000.0):
        continue
    hits.append({"addr": ramdiff.BASE + 4*i, "vals": [x/65536.0 for x in v],
                 "px_per_frame": per_frame})

print(f"\n{len(hits)} enderecos congelam nos repousos e correm durante a caminhada:")
for r in sorted(hits, key=lambda r: -abs(r["px_per_frame"]))[:20]:
    vs = "  ".join(f"{x:9.2f}" for x in r["vals"])
    print(f"  0x{r['addr']:08X}  {vs}   {r['px_per_frame']:+.3f} px/frame")

json.dump(hits[:200], open(os.path.join(HERE, "guided2_hits.json"), "w"), indent=1)
print("\nordem das colunas: [parado, parado, anda, anda, parado, parado]")
print("salvo em guided2_hits.json")
