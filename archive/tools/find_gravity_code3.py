"""Busca corrigida: capstone com skipdata, comecando no ENTRY POINT do ARM9.

Erro da versao anterior: capstone interrompe o gerador na primeira instrucao
invalida. O ARM9 comeca com 0x800 bytes de preenchimento (E7FFDEFF) porque o
entry point e 0x02000800, entao a desmontagem parava ali e os "zero hits" nao
significavam nada.

Controle embutido: contamos instrucoes decodificadas e hits de um offset banal
(#0x20). Se o controle falhar, o resto do resultado e descartavel.
"""
import struct, sys, collections
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD = rom.arm9RamAddress
ENTRY = rom.arm9EntryAddress
print(f"ARM9 {len(arm9):,} bytes em 0x{LOAD:08X}; entry 0x{ENTRY:08X}")
off_entry = ENTRY - LOAD
print(f"desmontando a partir de offset 0x{off_entry:X} (pula o preenchimento)\n")


def make(mode):
    md = Cs(CS_ARCH_ARM, mode | CS_MODE_LITTLE_ENDIAN)
    md.skipdata = True          # nao aborta em bytes invalidos
    return md


for nm, mode in (("THUMB", CS_MODE_THUMB), ("ARM", CS_MODE_ARM)):
    md = make(mode)
    total = 0
    control = 0
    hits = collections.defaultdict(list)
    for ins in md.disasm(arm9[off_entry:], ENTRY):
        if ins.mnemonic == ".byte":
            continue
        total += 1
        s = ins.op_str
        if "#0x20" in s or "#0x24" in s or "#0x28" in s:
            control += 1
        for tgt in ("#0xb4", "#0xb8", "#0xbc", "#0xc0", "#0xc4"):
            if tgt in s and "[" in s:
                hits[tgt].append((ins.address, f"{ins.mnemonic} {s}"))
    print(f"== modo {nm} ==")
    print(f"   instrucoes decodificadas: {total:,}")
    print(f"   controle (#0x20/#0x24/#0x28): {control:,}  "
          f"{'OK' if control > 100 else '<<< CONTROLE FALHOU: resultado nao confiavel'}")
    for tgt in ("#0xb4", "#0xb8", "#0xbc", "#0xc0", "#0xc4"):
        lst = hits.get(tgt, [])
        print(f"   acesso a {tgt}: {len(lst)}")
        for addr, txt in lst[:6]:
            print(f"        0x{addr:08X}: {txt}")
    print()
