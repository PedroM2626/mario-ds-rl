"""Busca correta no capstone: par ADD rX,#im8  +  LDR/STR rY,[rX,#disp].

Correcao do erro anterior: no Thumb do ARM946E-S (sem Thumb-2) o LDR de 16 bits
so alcancar deslocamento ate 0x7C. Entao acesso a actor+0xBC e SEMPRE emitido
como ADD rX,#im8 (imediato em BYTES, nao multiplicado por 4) seguido de
LDR/STR rY,[rX,#disp]. A varredura antiga escalava o imediato por 4 e por isso
nao achou nada.

Cobertura ja medida e validada: modo THUMB com skipdata a partir de 0x800
produz ~212k instrucoes com controle positivo forte.
"""
import struct, sys, collections
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD, ENTRY = rom.arm9RamAddress, rom.arm9EntryAddress

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_LITTLE_ENDIAN)
md.skipdata = True
insns = [i for i in md.disasm(arm9[ENTRY - LOAD:], ENTRY) if i.mnemonic != ".byte"]
print(f"instrucoes Thumb: {len(insns):,}")

TARGETS = {0xB4: "velH", 0xB8: "minVelH", 0xBC: "accelV(gravidade)",
           0xC0: "minVelV(terminal)", 0xC4: "accelH", 0xC8: "?_c8", 0xCC: "velocity"}

# indice por endereco para olhar vizinhos
pos = {ins.address: k for k, ins in enumerate(insns)}
found = collections.defaultdict(list)

for k, ins in enumerate(insns):
    m = ins.mnemonic
    # forma 1: deslocamento direto grande so existe em ARM; em Thumb 16-bit max 0x7C
    # forma 2: ADD rX, #im8  -> depois LDR/STR rY,[rX,#disp]
    if m == "add":
        ops = ins.op_str
        # "r3, #160"  ou "r3, r0, #4"
        parts = [p.strip() for p in ops.split(",")]
        imm = None
        base_reg = parts[0]
        if len(parts) == 2 and parts[1].startswith("#"):
            try:
                imm = int(parts[1].lstrip("#"), 0)
            except ValueError:
                imm = None
        elif len(parts) == 3 and parts[2].startswith("#"):
            try:
                imm = int(parts[2].lstrip("#"), 0)
            except ValueError:
                imm = None
        if imm is None or imm < 0x40:
            continue
        # procura LDR/STR usando esse registrador nos proximos 6 instrucoes
        for j in range(k + 1, min(k + 7, len(insns))):
            nxt = insns[j]
            if nxt.mnemonic not in ("ldr", "str", "ldrb", "strb", "ldrh", "strh"):
                continue
            o = nxt.op_str
            if "[" + base_reg + "," not in o and not o.endswith("[" + base_reg + "]"):
                continue
            i2 = o.find("#")
            if i2 < 0:
                disp = 0
            else:
                try:
                    disp = int(o[i2:].rstrip("]"), 0)
                except ValueError:
                    continue
            total = imm + disp
            if total in TARGETS:
                found[total].append((nxt.address, f"{ins.mnemonic} {ins.op_str} ; {nxt.mnemonic} {nxt.op_str}"))
            break

print("\n== acessos a campos do Actor (offset efetivo = ADD + deslocamento) ==")
for off in sorted(TARGETS):
    lst = found.get(off, [])
    print(f"  +0x{off:03X} {TARGETS[off]:<22} {len(lst):>4} ocorrencias")
    for addr, txt in lst[:6]:
        print(f"        0x{addr:08X}: {txt}")
