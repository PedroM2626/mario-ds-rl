"""Refino: so acessos RELATIVOS A OBJETO em 0xB4..0xC4.

A varredura anterior mostrou que os hits eram quase todos [sp,#..] (variaveis
locais de pilha) e [pc,#..] (literal pool). O que procuramos e [rN,#0xbc] com rN
sendo um ponteiro de objeto -- a forma como accelV/minVelV/accelH sao lidos
dentro de Actor::updateVerticalVelocity e cia.

Depois, procuramos o par (0xBC, 0xC0) a curta distancia: velocity.y += accelV
com clamp por minVelV usa os dois na mesma funcao.
"""
import re, sys, collections
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD, ENTRY = rom.arm9RamAddress, rom.arm9EntryAddress
off_entry = ENTRY - LOAD

OBJ = re.compile(r"\[(r\d+|sl|fp),\s*#(0xb4|0xb8|0xbc|0xc0|0xc4)\]")

for nm, mode in (("THUMB", CS_MODE_THUMB), ("ARM", CS_MODE_ARM)):
    md = Cs(CS_ARCH_ARM, mode | CS_MODE_LITTLE_ENDIAN)
    md.skipdata = True
    ins_list = []
    for ins in md.disasm(arm9[off_entry:], ENTRY):
        if ins.mnemonic == ".byte":
            continue
        ins_list.append(ins)
    print(f"== modo {nm}: {len(ins_list):,} instrucoes ==")

    by_off = collections.defaultdict(list)
    for ins in ins_list:
        m = OBJ.search(ins.op_str)
        if m and "sp" not in m.group(1) and "pc" not in m.group(1):
            by_off[m.group(2)].append((ins.address, f"{ins.mnemonic} {ins.op_str}"))

    for tgt in ("0xb4", "0xb8", "0xbc", "0xc0", "0xc4"):
        lst = by_off.get(tgt, [])
        print(f"   acesso a objeto +{tgt}: {len(lst)}")
        for addr, txt in lst[:8]:
            print(f"        0x{addr:08X}: {txt}")

    # par 0xBC + 0xC0 proximos: assinatura da integracao de gravidade
    print("   -- pares +0xBC e +0xC0 a menos de 0x30 bytes --")
    bc = [a for a, _ in by_off.get("0xbc", [])]
    c0 = [a for a, _ in by_off.get("0xc0", [])]
    shown = 0
    for a in bc:
        near = [b for b in c0 if abs(b - a) < 0x30]
        if near:
            lo = min([a] + near) - 0x18
            txts = [(i.address, f"{i.mnemonic} {i.op_str}")
                    for i in ins_list if lo <= i.address <= max([a] + near) + 0x18]
            print(f"        bloco em 0x{lo:08X}:")
            for ad, t in txts:
                mark = "  <<<" if OBJ.search(t.split(" ", 1)[-1]) else ""
                print(f"           0x{ad:08X}: {t}{mark}")
            shown += 1
            if shown >= 3:
                break
    if shown == 0:
        print("        (nenhum par encontrado)")
    print()
