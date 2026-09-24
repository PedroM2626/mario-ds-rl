"""Segue a cadeia de ponteiros em vez de chutar o slot.

Os vizinhos de 0x0209DFxx sao valores de ~34 milhoes, que e o intervalo da RAM
principal do ARM9. Entao aquilo e uma tabela de ponteiros para objetos. Este
script:
  1. le a janela e marca o que e ponteiro plausivel (0x02000000..0x02400000);
  2. segue cada ponteiro e imprime as primeiras palavras do objeto apontado;
  3. procura, dentro dos alvos, trios contiguos com cara de posicao em 2^20
     (inteiros pequenos em parte alta, subpixel variando na parte baixa).
Tudo offline sobre ram_fields.bin -- nenhuma coleta nova.
"""
import struct, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "ram_fields.bin")
if not os.path.exists(path):
    raise SystemExit("ram_fields.bin nao existe; rode read_actor_fields.py antes")
blob = open(path, "rb").read()
RAM_LO, RAM_HI = 0x02000000, 0x02400000

def word(addr):
    o = addr - RAM_LO
    if 0 <= o <= len(blob) - 4:
        return struct.unpack_from("<i", blob, o)[0]
    return None

def uword(addr):
    o = addr - RAM_LO
    if 0 <= o <= len(blob) - 4:
        return struct.unpack_from("<I", blob, o)[0]
    return None

WIN_LO, WIN_HI = 0x0209DF00, 0x0209E100
print(f"== janela 0x{WIN_LO:08X}..0x{WIN_HI:08X}: o que e ponteiro valido? ==")
ptrs = []
for a in range(WIN_LO, WIN_HI, 4):
    v = uword(a)
    if v is None:
        continue
    if RAM_LO <= v < RAM_HI and (v & 3) == 0:
        ptrs.append((a, v))
print(f"{len(ptrs)} palavras na janela sao ponteiros alinhados para a RAM do ARM9")
for a, v in ptrs[:20]:
    print(f"  [0x{a:08X}] -> 0x{v:08X}")

print("\n== objetos apontados: primeiras 12 palavras (bruto / em escala 2^20) ==")
seen = set()
for a, v in ptrs:
    if v in seen:
        continue
    seen.add(v)
    vals = [word(v + 4*k) for k in range(12)]
    if any(x is None for x in vals):
        continue
    px = [f"{x/2**20:+.2f}" for x in vals]
    print(f"  0x{v:08X} (de 0x{a:08X}):")
    print(f"     bruto: " + " ".join(f"{x:>11}" for x in vals))
    print(f"     /2^20: " + " ".join(f"{s:>11}" for s in px))
    if len(seen) >= 10:
        break
