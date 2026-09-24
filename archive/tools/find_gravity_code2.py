"""Busca via capstone, em ARM e Thumb, por acesso indireto a 0xBC/0xC0.

A versao anterior mascarou encoding ARM a mao e nao casou nada: codigo de jogo
DS e em grande parte Thumb, e o LDR de 16 bits so alcancar offset ate 0x7C,
entao 0xBC aparece como instrugao larga de 32 bits ou como ADD+LDR.
Aqui desmontamos nos dois modos e procuramos pelo texto do operando, que cobre
as duas formas. Desync de desmontagem linear e esperado; agrupamentos coerentes
e o que importa.
"""
import struct, sys, collections
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD = rom.arm9RamAddress

# 1) busca direta por texto do operando nos dois modos
for mode_name, mode in (("ARM", CS_MODE_ARM), ("THUMB", CS_MODE_THUMB)):
    md = Cs(CS_ARCH_ARM, mode | CS_MODE_LITTLE_ENDIAN)
    hits = collections.Counter()
    where = collections.defaultdict(list)
    for ins in md.disasm(arm9, LOAD):
        s = ins.op_str
        if "#0xbc" in s or "#0xc0" in s:
            key = f"{ins.mnemonic} {s}"
            hits[key] += 1
            where[key].append(ins.address)
    print(f"== modo {mode_name}: acessos a 0xBC/0xC0 ==")
    if not hits:
        print("   (nada)")
    for k, c in hits.most_common(12):
        print(f"   {c:>4}x  {k:<40} ex 0x{where[k][0]:08X}")
    print()

# 2) padrao Thumb: ADD rX, #0xA0 ou #0xB0 seguido de LDR rY,[rX,#...]
#    (forma como o compilador chega em offsets acima de 0x7C)
print("== Thumb: ADD imediata 0x80..0xC0 seguida de LDR/STR (atinge 0xBC) ==")
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_LITTLE_ENDIAN)
found = 0
i = 0
while i < len(arm9) - 4:
    w = struct.unpack_from("<H", arm9, i)[0]
    # ADD rX, #imm5 (0x3000-0x37FF)
    if (w & 0xF800) == 0x3000:
        imm = w & 0x1F
        if imm in (0x20, 0x24, 0x28, 0x2C):     # 0x80,0xA0,0xB0,0xB0... x4
            nxt = struct.unpack_from("<H", arm9, i + 2)[0]
            if (nxt & 0xF800) in (0x6800, 0x6000):
                kind = "LDR" if (nxt & 0xF800) == 0x6800 else "STR"
                off = ((nxt >> 6) & 0x1F) * 4
                total = imm * 4 + off
                if total in (0xBC, 0xC0, 0xB8, 0xB4, 0xC4):
                    print(f"   0x{LOAD+i:08X}: ADD r{w&7}, #{imm*4:#x} ; {kind} .. #{off:#x}"
                          f"  -> offset efetivo {total:#x}")
                    found += 1
    i += 2
print(f"   total: {found}")
