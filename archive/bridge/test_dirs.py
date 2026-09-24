"""Teste trifasico com auto-verificacao de que o input chegou ao jogo.

Amostra o endereco candidato em tres janelas: parado / direita / esquerda.
Se a fase "direita" nao mover nada, o teste declara-se invalido em vez de
interpretar o silencio do endereco (foi o erro da versao anterior).
"""
import ctypes, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

user32 = ctypes.windll.user32
VK_RIGHT, VK_LEFT, VK_PAUSE = 0x27, 0x25, 0x13
UP = 0x0002
FIX = 65536.0
CAND = 0x020A7678

b = nsmb_env.Bridge()
CMD, ACKF = os.path.join(HERE, "cmd"), os.path.join(HERE, "ack")

hwnd = user32.FindWindowW(None, "DeSmuME 0.9.13 x64 SSE2 | New Super Mario Bros.")
print("[janela]", "encontrada" if hwnd else "NAO encontrada (titulo mudou?)")
if hwnd:
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)


def start_sample(tag, n, addrs):
    if os.path.exists(ACKF):
        os.remove(ACKF)
    with open(CMD + ".tmp", "w") as f:
        f.write("sample %s %d %s" % (tag, n, " ".join(hex(a) for a in addrs)))
    os.replace(CMD + ".tmp", CMD)


def wait_done(timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(ACKF) and "sample done" in open(ACKF, errors="ignore").read():
            os.remove(ACKF)
            return True
        time.sleep(0.1)
    return False


def series(tag):
    path = os.path.join(HERE, "sample_%s.csv" % tag)
    out = []
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) == 2:
                out.append(int(p[1]))
    return out


def phase(tag, n, key=None):
    start_sample(tag, n, [CAND])
    time.sleep(0.25)
    if key:
        user32.keybd_event(key, 0, 0, 0)
        time.sleep(0.05 * n + 1.0)
        user32.keybd_event(key, 0, UP, 0)
    ok = wait_done()
    s = series(tag) if ok else []
    if not s:
        return None
    px = [v / FIX for v in s]
    return {"n": len(s), "first": px[0], "last": px[-1], "delta": px[-1] - px[0],
            "min": min(px), "max": max(px)}


print("\n[aviso] se a fase 'direita' nao mover, o teste e invalido.")

idle = phase("f_idle", 90)
print("parado  :", idle)
right = phase("f_right", 120, VK_RIGHT)
print("direita :", right)
left = phase("f_left", 120, VK_LEFT)
print("esquerda:", left)

if not right or abs(right["delta"]) < 1.0:
    print("\n>>> TESTE INVALIDO: o jogador nao se moveu na fase 'direita'.")
    print("    Causas provaveis: emulacao pausada, ou foco nao esta no DeSmuME.")
    print("    Nada pode ser concluido sobre o endereco a partir daqui.")
    sys.exit(2)

print("\nMovimento confirmado (%.2f px na direita). Interpretacao:" % right["delta"])
if left and abs(left["delta"]) > 1.0:
    if left["delta"] < 0 and right["delta"] > 0:
        print("  direita aumenta e esquerda diminui -> 0x%08X e POSICAO X com sinal." % CAND)
    else:
        print("  ambos aumentam -> acumulador, nao posicao.")
else:
    print("  direita move mas esquerda NAO move -> endereco sensivel so a um dos")
    print("  sentidos: suspeito de ser camera/scroll ou um contador de progresso.")
