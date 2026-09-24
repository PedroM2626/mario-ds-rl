"""Decisiones sobre o endereco candidato 0x020A7678.

Usa o amostrador por frame do Lua (comando "sample") gravando durante:
  (1) 90 frames ocioso  -> deve ficar constante
  (2) 150 frames andando para a ESQUERDA -> se cair, e posicao com sinal
Ate' entao o Lua nao tem API de input neste build, entao as teclas sao fisicas.
"""
import ctypes, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

user32 = ctypes.windll.user32
VK_LEFT, VK_RIGHT = 0x25, 0x27
UP = 0x0002
CAND = 0x020A7678
FIX = 65536.0

b = nsmb_env.Bridge()
print("[ping]", b.send("ping", timeout=15))

hwnd = user32.FindWindowW(None, "DeSmuME 0.9.13 x64 SSE2 | New Super Mario Bros.")
user32.SetForegroundWindow(hwnd)
time.sleep(0.5)

CMD = os.path.join(HERE, "cmd")
ACKF = os.path.join(HERE, "ack")


def start_sample(tag, nframes, addrs):
    if os.path.exists(ACKF):
        os.remove(ACKF)
    hexs = " ".join(hex(a) for a in addrs)
    with open(CMD + ".tmp", "w") as f:
        f.write(f"sample {tag} {nframes} {hexs}")
    os.replace(CMD + ".tmp", CMD)


def wait_sample(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(ACKF):
            txt = open(ACKF, errors="ignore").read().strip()
            if "sample done" in txt:
                os.remove(ACKF)
                return True
        time.sleep(0.1)
    return False


def read_csv(tag):
    path = os.path.join(HERE, f"sample_{tag}.csv")
    rows = []
    with open(path) as f:
        head = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == len(head):
                rows.append((int(parts[0]), [int(x) for x in parts[1:]]))
    return head, rows


def summarize(tag, head, rows):
    print(f"\n--- {tag}: {len(rows)} amostras; colunas {head[1:]}")
    if not rows:
        print("  VAZIO"); return
    for ci in range(len(rows[0][1])):
        v = [r[1][ci] for r in rows]
        px = [x / FIX for x in v]
        d = [(b_ - a_) / FIX for a_, b_ in zip(px, px[1:])]
        print(f"  {head[1+ci]}: inicio={px[0]:.4f} fim={px[-1]:.4f} min={min(px):.4f} max={max(px):.4f}"
              f"  delta_total={px[-1]-px[0]:+.4f} px"
              f"  |passo| max={max((abs(x) for x in d), default=0):.4f}")


# (1) ocioso
start_sample("idle", 90, [CAND])
time.sleep(2.5)
assert wait_sample(30), "timeout amostra ociosa"
summarize("ocioso", *read_csv("idle"))

# (2) andando para a esquerda
start_sample("left", 150, [CAND])
time.sleep(0.3)
user32.keybd_event(VK_LEFT, 0, 0, 0)
time.sleep(2.6)
user32.keybd_event(VK_LEFT, 0, UP, 0)
assert wait_sample(40), "timeout amostra esquerda"
head, rows = read_csv("left")
summarize("andando a esquerda", head, rows)

vals = [r[1][0] / FIX for r in rows]
trend = vals[-1] - vals[0]
print(f"\ninterpretagao: {'DIMINUI -> coordenada com sinal (posicao)' if trend < -1 else 'NAO diminui -> provavelmente acumulador ou outra grandeza'}")
