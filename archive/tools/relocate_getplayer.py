"""Relocaliza Game::getPlayer no ROM EUROPEU (A2DP) por assinatura de bytes.

O mapa de simbolos CC0 e do build USA (A2DE); enderecos nao transferem. Mas a
funcao tem 0x10 bytes e uma forma inconfundivel: carrega um global via literal
pool, indexa por r0<<2 e retorna. Procuramos a sequencia de instrucoes, nao o
endereco.
"""
import struct
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD = rom.arm9RamAddress

# ldr r0,[r3,r0,lsl #2]  == E7930100   ;   bx lr == E12FFF1E
NEEDLE = struct.pack("<II", 0xE7930100, 0xE12FFF1E)
print(f"buscando assinatura {NEEDLE.hex()} em {len(arm9):,} bytes\n")

md = Cs(CS_ARCH_ARM, CS_MODE_ARM | CS_MODE_LITTLE_ENDIAN)
hits = []
pos = 0
while True:
    i = arm9.find(NEEDLE, pos)
    if i < 0:
        break
    pos = i + 4
    if i % 4:
        continue
    # a funcao de 0x10 bytes comecaria 2 instrucoes antes
    start = i - 8
    if start < 0:
        continue
    code = arm9[start:start + 0x10]
    ins = list(md.disasm(code, LOAD + start))
    if len(ins) != 4:
        continue
    txt = [f"{x.mnemonic} {x.op_str}".strip() for x in ins]
    # exigimos o padrao: ldr r3,[pc,#N] / (qualquer) / ldr r0,[r3,r0,lsl#2] / bx lr
    if not (txt[0].startswith("ldr") and "pc" in txt[0]):
        continue
    lit_off = int(txt[0].split("#")[-1].rstrip("]")) if "#" in txt[0] else None
    lit = None
    if lit_off is not None:
        la = (start + 8 + lit_off + 8) & ~3      # PC = instr+8
        if 0 <= la < len(arm9) - 3:
            lit = struct.unpack_from("<I", arm9, la)[0]
    hits.append((LOAD + start, txt, lit))

print(f"{len(hits)} candidatos com a assinatura de indexador de array:\n")
for addr, txt, lit in hits[:12]:
    l = f"  -> global 0x{lit:08X}" if lit and 0x02000000 <= lit < 0x02400000 else ""
    print(f"  0x{addr:08X}: " + " | ".join(txt) + l)
