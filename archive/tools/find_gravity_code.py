"""Busca ancorada em ENCODING no ARM9 europeu: acesso aos offsets 0xBC/0xC0.

Ancora: Actor.hpp declara velH, minVelH, accelV, minVelV, accelH(0xC4).
Logo accelV = +0xBC e minVelV = +0xC0. updateVerticalVelocity e
velocity.y += accelV com clamp por minVelV, entao o codigo real tem que carregar
de [rN,#0xBC] e [rN,#0xC0] perto um do outro.

Nao dependemos de sincronia de desmontagem linear: varremos o binario como
palavras de 32 bits e casamos o encoding ARM de LDR/STR com immediato 0xBC/0xC0
diretamente. Depois desmontamos o contexto de cada agrupamento.
"""
import struct, sys
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD = rom.arm9RamAddress
print(f"ARM9 {len(arm9):,} bytes em 0x{LOAD:08X}")

# ARM: cond 0101 L P U B W L Rn Rt imm12
#  LDR Rt,[Rn,#imm] : bits 27:25=101, 22(P)=1,21(U)=1,20(B)=0,18(L)=1
#  STR Rt,[Rn,#imm] : idem com bit 18 = 0
MASK = 0x0FFFFF00          # ignora Rt (bits 13:10) e preserva imm12 alto? nao:
# imm12 = bits 11:0 -> queremos casar 0xBC exatamente; Rt livre em 13:10
M = 0x00000F00             # mascara sobre Rt
def base_pat(is_ldr, imm):
    v = 0x05100000 | (1 << 22) | (1 << 21) | (1 << 20) | (imm & 0xFFF)
    if is_ldr:
        v |= (1 << 18)
    return v

LDR_BC = base_pat(True, 0xBC)
STR_BC = base_pat(False, 0xBC)
LDR_C0 = base_pat(True, 0xC0)
STR_C0 = base_pat(False, 0xC0)
CMPMASK = 0x0000F00 | 0x00FFF000 ^ 0x00FFF000  # placeholder, clareza abaixo
# Mascara efetiva: cond livre (bits 31:28), Rn livre (17:14), Rt livre (13:10)
MASK_EFF = 0x00000000 | (0xF << 28) | (0xF << 14) | (0xF << 10)
MASK_EFF = ~MASK_EFF & 0xFFFFFFFF

def find(pat):
    out = []
    for i in range(0, len(arm9) - 3, 4):
        w = struct.unpack_from("<I", arm9, i)[0]
        if (w & MASK_EFF) == (pat & MASK_EFF):
            out.append(LOAD + i)
    return out

hits = {n: find(p) for n, p in [("LDR #0xBC", LDR_BC), ("STR #0xBC", STR_BC),
                                 ("LDR #0xC0", LDR_C0), ("STR #0xC0", STR_C0)]}
for k, v in hits.items():
    print(f"{k:<12} {len(v):>4} ocorrencias" + (f"  ex: {[hex(x) for x in v[:6]]}" if v else ""))

# agrupamentos onde LDR 0xBC e LDR 0xC0 aparecem proximos: assinatura de
# updateVerticalVelocity
print("\n== pares 0xBC e 0xC0 a menos de 0x40 bytes (assinatura da integracao) ==")
clusters = []
for a in hits["LDR #0xBC"]:
    near = [b for b in hits["LDR #0xC0"] + hits["STR #0xC0"] if abs(b - a) < 0x40]
    if near:
        clusters.append((a, near))
seen = set()
for a, near in clusters:
    key = min([a] + near) // 0x40
    if key in seen:
        continue
    seen.add(key)
    lo = min([a] + near) - 0x10
    hi = max([a] + near) + 0x20
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM | CS_MODE_LITTLE_ENDIAN)
    print(f"\n  bloco 0x{lo:08X}..0x{hi:08X}")
    for ins in md.disasm(arm9[lo - LOAD:hi - LOAD], lo):
        mark = "  <<<" if ins.address in [a] + near else ""
        print(f"    0x{ins.address:08X}: {ins.mnemonic:<7} {ins.op_str}{mark}")
    if len(seen) >= 6:
        break
