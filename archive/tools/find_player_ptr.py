"""Desmonta Game::getPlayer(0x02020608) no ARM9 para achar o ponteiro global do
ator do jogador.

Motivo: localizar o Mario na RAM por desmontagem e deterministico, enquanto
procurar por diferenca de valor e heuristico e dependente de captura humana.
Endereco de simbolo vem do mapa CC0 config/A2DE/arm9/symbols.txt.
"""
import struct, sys
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD = rom.arm9RamAddress          # 0x02000000
print(f"ARM9: {len(arm9):,} bytes carregados em 0x{LOAD:08X}")

TARGET = 0x02020608                # Game::getPlayer, tamanho 0x10 (mapa A2DE/USA)
off = TARGET - LOAD
if not (0 <= off < len(arm9)):
    sys.exit(f"0x{TARGET:08X} fora do ARM9")

md = Cs(CS_ARCH_ARM, CS_MODE_ARM | CS_MODE_LITTLE_ENDIAN)
md.detail = True
print(f"\n== desmontagem em 0x{TARGET:08X} ==")
lits = []
for ins in md.disasm(arm9[off:off + 0x40], TARGET):
    print(f"  0x{ins.address:08X}:  {ins.bytes.hex():<10} {ins.mnemonic} {ins.op_str}")
    # literais de LDR pc-relative aparecem logo depois do codigo
    if ins.mnemonic.startswith("ldr") and "[pc" in ins.op_str:
        lits.append(ins.address)

print("\n== palavras da area (para resolver o literal pool) ==")
for i in range(0, 0x40, 4):
    w = struct.unpack_from("<I", arm9, off + i)[0]
    print(f"  0x{TARGET+i:08X}: 0x{w:08X}", "  <- parece ponteiro para RAM do ARM9"
          if 0x02000000 <= w < 0x02400000 else "")
