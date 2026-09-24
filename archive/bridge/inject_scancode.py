"""Injeta teclas com SCANCODE via SendInput, para DirectInput enxergar.

Hipotese: as tentativas anteriores usaram keybd_event com apenas VK. Aplicativos
que consultam o teclado via DirectInput (o DeSmuME faz isso) frequentemente nao
veem eventos sinteticos sem scancode. Aqui usamos SendInput com
KEYEVENTF_SCANCODE e, para as setas, o flag de tecla estendida (0xE0).
"""
import ctypes, sys, time
from ctypes import wintypes

user32 = ctypes.windll.user32

INPUT_KEYBOARD = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001

# scancodes (set 1): direita = 0x4D estendida, esquerda = 0x4B, A(x) = 0x2D, B(z) = 0x2C
SCAN = {"right": (0x4D, True), "left": (0x4B, True), "up": (0x48, True),
        "down": (0x50, True), "A": (0x2D, False), "B": (0x2C, False), "start": (0x1C, False)}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def send(scan, extended, up):
    extra = ctypes.c_ulong(0)
    i = INPUT()
    i.type = INPUT_KEYBOARD
    i.ki = KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE | (0xE000 if extended else 0)
                     | (KEYEVENTF_KEYUP if up else 0), 0, ctypes.pointer(extra))
    # O DeSmuME checa extended via flag separado; mandamos ambos.
    i.ki.dwFlags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0) \
                   | (KEYEVENTF_KEYUP if up else 0)
    return user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(INPUT))


def hold(key, seconds):
    scan, ext = SCAN[key]
    n = send(scan, ext, False)
    time.sleep(seconds)
    m = send(scan, ext, True)
    return n, m


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "right"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

    sys.path.insert(0, ".")
    import nsmb_env
    b = nsmb_env.Bridge()
    st = b.status()
    print("[status]", st, flush=True)
    if st.get("emulating") != 1:
        raise SystemExit("ABORTADO: jogo nao esta emulando.")

    hwnd = user32.FindWindowW(None, "DeSmuME 0.9.13 x64 SSE2 | New Super Mario Bros.")
    print(f"[foco] hwnd={hwnd} foreground={'sim' if user32.GetForegroundWindow() == hwnd else 'NAO'}")
    if hwnd:
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.5)

    f0 = int(b.status()["framecount"])
    print(f"[injetando] segurar '{key}' por {secs}s com scancode (framecount {f0})", flush=True)
    down, up = hold(key, secs)
    f1 = int(b.status()["framecount"])
    print(f"[injetando] SendInput retornou down={down} up={up}; frames {f0} -> {f1}")
    print(">>> tire uma screenshot agora e compare com a anterior <<<")
