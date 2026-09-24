"""Captura o par de dumps real e localiza a posicao do jogador.

Mantem a tecla Direita fisicamente pressionada (SendInput) enquanto o DeSmuME
roda, despeja a RAM antes e depois, e roda a busca diferencial.
Tambem garante que a emulacao nao esta pausada comparando o contador de frames
do trace.csv.
"""
import ctypes, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env, ramdiff

user32 = ctypes.windll.user32
VK_RIGHT, VK_PAUSE = 0x27, 0x13
KEYEVENTF_KEYUP = 0x0002


def trace_frame():
    p = os.path.join(HERE, "trace.csv")
    if not os.path.exists(p):
        return -1
    with open(p, "rb") as f:
        f.seek(max(0, os.path.getsize(p) - 4096))
        tail = f.read().decode("ascii", "ignore").strip().splitlines()
    for line in reversed(tail):
        n = line.split(",")[0].strip()
        if n.isdigit():
            return int(n)
    return -1


def ensure_running(b):
    a = trace_frame()
    time.sleep(1.0)
    c = trace_frame()
    print(f"[frames] {a} -> {c}")
    if c <= a:
        print("[frames] emulacao parada; enviando tecla Pause")
        user32.keybd_event(VK_PAUSE, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(VK_PAUSE, 0, KEYEVENTF_KEYUP, 0, 0)
        time.sleep(1.0)
        print(f"[frames] agora em {trace_frame()}")
    return c


def hold(seconds, key=VK_RIGHT):
    user32.keybd_event(key, 0, 0, 0)
    time.sleep(seconds)
    user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0, 0)


b = nsmb_env.Bridge()
print("[ping]", b.send("ping", timeout=10))
ensure_running(b)

# foco no emulador para as teclas chegarem nele
hwnd = user32.FindWindowW(None, "DeSmuME 0.9.13 x64 SSE2 | New Super Mario Bros.")
if hwnd:
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.4)

A = b.dump("walk_a", 0x02000000, 0x400000)
print(f"[dump A] {len(A):,} bytes, frame {trace_frame()}")
hold(2.5)
B = b.dump("walk_b", 0x02000000, 0x400000)
print(f"[dump B] {len(B):,} bytes, frame {trace_frame()}")

diff = sum(1 for i in range(0, len(A), 4) if A[i:i+4] != B[i:i+4])
print(f"[diff] {diff:,} palavras de 32 bits mudaram ({diff/ (len(A)//4):.2%} da RAM)")

c = ramdiff.candidates_pair(A, B, direction="right")
print(f"\n{len(c)} candidatos a coordenada X (movimento para a direita)")
for r in c[:20]:
    print(f"  0x{r['addr']:08X}  {r['before']/ramdiff.FIX:>12.4f} -> {r['after']/ramdiff.FIX:>12.4f}"
          f"   d={r['delta_px']:+10.4f} px")
