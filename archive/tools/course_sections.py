"""Decodifica o layout de secoes de um arquivo de nivel do NSMB DS.

Header = u32 tamanho do cabecalho, seguido de pares (offset, length) que
segmentam todo o resto do arquivo.
"""
import struct, sys
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
name = sys.argv[1] if len(sys.argv) > 1 else "course/A01_1.bin"
blob = rom.getFileByName(name)
hdr = struct.unpack_from("<I", blob, 0)[0]
print(f"{name}: {len(blob):,} bytes, cabecalho de 0x{hdr:X}\n")

secs = []
base = struct.unpack_from("<I", blob, 4)[0]   # u32 sozinho em 0x04
print(f"u32 @0x04 = 0x{base:X} (inicio do primeiro bloco de dados)")
for off in range(8, hdr - 7, 8):
    o, ln = struct.unpack_from("<II", blob, off)
    secs.append((off, o, ln))

print(f"\n{'@hdr':>6} {'offset':>8} {'len':>7} {'fim':>8} {'x8B':>6} {'x16B':>6}  status")
cur = base
for ho, o, ln in secs:
    end = o + ln
    tag = "OK" if end <= len(blob) else "ESTOUROU"
    if o != cur:
        tag += f" (gap/lixo de {o-cur:+d})"
    cur = end
    print(f"  0x{ho:04X}  0x{o:06X} {ln:>7}  0x{end:06X} {ln/8:>6.1f} {ln/16:>6.1f}  {tag}")
print(f"\nimagem: secoes vao de 0x{base:X} a 0x{cur:X}; arquivo tem 0x{len(blob):X}"
      f" {'-> fechamento exato' if cur == len(blob) else '-> sobra lixo'}\n")

# candidatos a lista de objetos: segmentos grandes multiplos de 8
for ho, o, ln in secs:
    if ln < 32 or ln % 8:
        continue
    body = blob[o:o+ln]
    n = ln // 8
    recs = [struct.unpack_from("<4H", body, i*8) for i in range(n)]
    f0 = [r[0] for r in recs]; f1 = [r[1] for r in recs]
    f2 = [r[2] for r in recs]; f3 = [r[3] for r in recs]
    mono = sum(1 for a, b in zip(f1, f1[1:]) if b > a)
    print(f"== 0x{o:06X} len {ln} ({n} x 8B) ==")
    print(f"   w0: min {min(f0)} max {max(f0)} unicos {len(set(f0))}")
    print(f"   w1: min {min(f1)} max {max(f1)} monotono-crescente em {mono}/{n-1}")
    print(f"   w2: min {min(f2)} max {max(f2)} unicos {len(set(f2))}")
    print(f"   w3: min {min(f3)} max {max(f3)} unicos {len(set(f3))}")
    for i in range(min(n, 12)):
        a, b, c, d = recs[i]
        print(f"     {i:>3}  id=0x{a:04X} x={b:<6} y={c:<6} p=0x{d:04X}")
    print()
