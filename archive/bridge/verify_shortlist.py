"""Verificacao decisiva da lista curta: amostra a 60 Hz enquanto segura Direita.

Se o endereco for mesmo a posicao X do jogador, a serie vai subir de forma
sucessiva e continua, e a inclinacao nos da a ESCALA real (px/frame), que os
despejos de 3 em 3 segundos nao conseguiam medir.
"""
import ctypes, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

user32 = ctypes.windll.user32
VK_RIGHT, VK_PAUSE = 0x27, 0x13
UP = 0x0002
FIX = 65536.0
CMD, ACKF = os.path.join(HERE, "cmd"), os.path.join(HERE, "ack")

SHORTLIST = [0x02098238, 0x0209823C, 0x02098240, 0x02098244,
             0x020A7678, 0x021C1D00, 0x021C2A64, 0x020A59BC]

b = nsmb_env.Bridge()
print("[ping]", b.send("ping", timeout=15))

hwnd = user32.FindWindowW(None, "DeSmuME 0.9.13 x64 SSE2 | New Super Mario Bros.")
user32.SetForegroundWindow(hwnd)
time.sleep(0.6)

NF = 420   # ~7 segundos a 60 Hz
if os.path.exists(ACKF):
    os.remove(ACKF)
with open(CMD + ".tmp", "w") as f:
    f.write("sample verify %d %s" % (NF, " ".join(hex(a) for a in SHORTLIST)))
os.replace(CMD + ".tmp", CMD)
print(f"[sample] {NF} frames em {len(SHORTLIST)} enderecos; segurando Direita...")

time.sleep(0.6)
user32.keybd_event(VK_RIGHT, 0, 0, 0)
time.sleep(5.0)
user32.keybd_event(VK_RIGHT, 0, UP, 0)
print("[input] Direita solta; aguardando fim da amostragem")

t0 = time.time()
while time.time() - t0 < 90:
    if os.path.exists(ACKF) and "sample done" in open(ACKF, errors="ignore").read():
        os.remove(ACKF)
        break
    time.sleep(0.2)
else:
    print("TIMEOUT esperando sample done")
    sys.exit(1)

path = os.path.join(HERE, "sample_verify.csv")
head = open(path).readline().strip().split(",")[1:]
data = np.genfromtxt(path, delimiter=",", skip_header=1)
fr = data[:, 0]
print(f"\n{len(fr)} amostras; frames {fr[0]:.0f}..{fr[-1]:.0f}")

for i, a in enumerate(SHORTLIST):
    col = data[:, i + 1] / FIX
    d = np.diff(col)
    live = col[20:320]          # janela em que Direita estava pressionada
    idle = np.concatenate([col[:15], col[360:]])
    print(f"\n  0x{a:08X}: bruto[{col.min():.2f}..{col.max():.2f}] "
          f"durante_direita: d_medio={np.mean(np.diff(live)) if len(live)>1 else 0:+.4f} px/frame"
          f"  parado: d_medio={np.mean(np.abs(np.diff(idle))) if len(idle)>1 else 0:.4f}")
    print("     amostras:", " ".join(f"{v:.1f}" for v in col[::40]))
