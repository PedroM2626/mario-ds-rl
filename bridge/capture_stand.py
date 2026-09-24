"""Terceiro dump, agora SEM apertar nada, e intersecao dos tres criterios.

Criterio de exclusao forte: um endereco que e a posicao do jogador para de mudar
quando o jogador para. Coisa que continua mudando com o jogo ocioso e timer,
animacao de background, contagem de frames ou ruido -- nao e coordenada.
"""
import ctypes, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env, ramdiff

user32 = ctypes.windll.user32
VK_RIGHT, VK_PAUSE = 0x27, 0x13
UP = 0x0002
FIX = ramdiff.FIX


b = nsmb_env.Bridge()


def frames():
    """Relogio do EMULADOR, nao o contador do loop do Lua.

    Usar o contador do loop ja me deixou ler pausa como se fosse 'endereco
    congelado'. framecount so avanca com o jogo rodando de fato.
    """
    try:
        return int(b.status().get("framecount", -1))
    except Exception:
        return -1


def focus():
    hwnd = user32.FindWindowW(None, "DeSmuME 0.9.13 x64 SSE2 | New Super Mario Bros.")
    if hwnd:
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.4)
    return hwnd


b = nsmb_env.Bridge()
focus()
if not b.wait_running(1.5):
    raise SystemExit("ABORTADO: o emulador nao esta avancando frames (jogo pausado?). "
                     "Nada seria medido aqui.")

# A: parado antes  |  B: andando  |  C: parado depois
A = b.dump("s_a", 0x02000000, 0x400000); fa = frames()
time.sleep(1.5)
B0 = b.dump("s_b0", 0x02000000, 0x400000); fb0 = frames()

user32.keybd_event(VK_RIGHT, 0, 0, 0)
time.sleep(3.0)
user32.keybd_event(VK_RIGHT, 0, UP, 0)
W = b.dump("s_walk", 0x02000000, 0x400000); fw = frames()

time.sleep(1.5)
C = b.dump("s_c", 0x02000000, 0x400000); fc = frames()

print(f"frames: paradoA={fa} paradoB={fb0} aposAndar={fw} paradoC={fc}")
gap_walk = fw - fb0
gap_idle = fc - fw
print(f"gap do trecho andando = {gap_walk} frames; gap ocioso = {gap_idle} frames\n")


def words(blob):
    import struct
    return struct.unpack_from(f"<{len(blob)//4}i", blob, 0)


wa, wb0, ww, wc = words(A), words(B0), words(W), words(C)
n = min(len(wa), len(wb0), len(ww), len(wc))

hits = []
for i in range(n):
    idle_before = wb0[i] - wa[i]                 # deve ser 0 (parado)
    moved = ww[i] - wb0[i]                       # deve ser positivo (andou p/ direita)
    idle_after = wc[i] - ww[i]                   # deve ser 0 (parou)
    if idle_before != 0 or idle_after != 0:
        continue
    if moved <= 0:
        continue
    px = moved / FIX
    if not (0.2 <= px / max(1, gap_walk) <= 12.0):   # velocidade plausivel por frame
        continue
    if not (-2000.0 <= wc[i] / FIX <= 60000.0):
        continue
    hits.append({"addr": ramdiff.BASE + 4*i, "value": wc[i], "px": wc[i]/FIX,
                 "moved_px": px, "px_per_frame": px/max(1, gap_walk)})

hits.sort(key=lambda r: -r["moved_px"])
print(f"{len(hits)} enderecos congelam quando parado, avancam quando anda, "
      f"e na velocidade certa:")
for r in hits[:20]:
    print(f"  0x{r['addr']:08X}  valor_final={r['px']:>12.4f} px   avancou={r['moved_px']:>9.3f} px"
          f"   ({r['px_per_frame']:.3f} px/frame)")

if hits:
    import json
    json.dump([{"addr": h["addr"], "scale": "16.16", "evidence": h} for h in hits[:40]],
              open(os.path.join(HERE, "candidates_x.json"), "w"), indent=1)
    print(f"\nsalvo em candidates_x.json")
