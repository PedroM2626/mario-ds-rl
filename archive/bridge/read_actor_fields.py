"""Le os campos de fisica do Actor a partir da base recem-estabelecida.

Base deduzida: position.x em 0x0209DFA4 (escala 1 px = 2^20). Em Actor.hpp os
campos vem em ordem velH, minVelH, accelV, minVelV, accelH(0xC4); se position e
o primeiro membro, esses campos caem em base+0xB8..0xC4 relativos ao X.
Lemos uma janela larga ao redor e interpretamos em tres escalas para nao
repetir o erro de embutir unidade no filtro.
"""
import os, sys
import ndspy.rom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsmb_env

X_ADDR = 0x0209DFA4
b = nsmb_env.Bridge()
st = b.status()
print("[status]", st)
if st.get("emulating") != 1:
    raise SystemExit("jogo nao esta emulando")

blob = b.dump("fields", 0x02000000, 0x400000)
import struct
def word(addr):
    o = addr - 0x02000000
    return struct.unpack_from("<i", blob, o)[0]

print(f"\nX em 0x{X_ADDR:08X} = {word(X_ADDR)}  ({word(X_ADDR)/2**20:.3f} px em escala 2^20)")
print(f"Y em 0x{X_ADDR+4:08X} = {word(X_ADDR+4)}")
print(f"Z em 0x{X_ADDR+8:08X} = {word(X_ADDR+8)}")

print("\n== janela de campos candidatos a fisica (offsets relativos ao X) ==")
print(f"{'offset':>8} {'endereco':<12} {'bruto':>14} {'/2^20':>10} {'/2^16':>10}")
for off in list(range(-0x20, 0x100, 4)):
    a = X_ADDR + off
    v = word(a)
    if v == 0:
        continue
    s16 = v / 65536.0
    s20 = v / (2**20)
    # constantes de fisica plausiveis: pequenos, com sinal, tipicamente < 0x10000
    mark = ""
    if 0 < abs(v) < 0x400000 and off >= 0xB0:
        mark = "  <-- plausivel como accel/velocidade terminal"
    print(f"{off:>+8} 0x{a:08X} {v:>14} {s20:>10.4f} {s16:>10.4f}{mark}")
