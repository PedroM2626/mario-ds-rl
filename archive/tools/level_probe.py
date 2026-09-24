"""Sonda o formato de um arquivo de nivel do NSMB DS.

Hipotese a testar: arquivo de course = cabecalho de 0x70 bytes + lista de
registros de 8 bytes (ID de objeto, X, Y, parametros).
"""
import struct, sys, collections
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())

name = sys.argv[1] if len(sys.argv) > 1 else "course/A01_1.bin"
blob = rom.getFileByName(name)
print(f"{name}: {len(blob):,} bytes")
print("magic u32 =", hex(struct.unpack_from("<I", blob, 0)[0]))
print()
print("== primeiros 0x70 bytes (cabecalho presumido) ==")
for off in range(0, 0x70, 16):
    row = blob[off:off+16]
    hexs = " ".join(f"{b:02x}" for b in row)
    u32s = " ".join(f"{struct.unpack_from('<I',blob,o)[0]:<10}" for o in range(off, min(off+16,len(blob)-3), 4))
    print(f"  {off:04X}  {hexs:<47}  {u32s}")

hdr = struct.unpack_from("<I", blob, 0)[0]
body = blob[hdr:]
print(f"\n== corpo a partir de 0x{hdr:X}: {len(body)} bytes -> {len(body)/8:.2f} registros de 8B, {len(body)/16:.2f} de 16B ==")

def stride_test(stride):
    n = len(body) // stride
    xs, ys, ids = [], [], []
    ok = True
    for i in range(n):
        r = body[i*stride:(i+1)*stride]
        w = struct.unpack_from("<" + "H"*(stride//2), r)
        ids.append(w[0]); xs.append(w[1]); ys.append(w[2])
    mono = sum(1 for a, b in zip(xs, xs[1:]) if b >= a)
    print(f"  stride {stride}: {n} registros | X cresce em {mono}/{max(1,n-1)} | "
          f"X em [0,{max(xs)}] Y em [0,{max(ys)}] IDs unicos {len(set(ids))}")
for s in (8, 16, 20, 24):
    stride_test(s)

print("\n== leitura como registros de 8 bytes (u16*4) ==")
recs = [struct.unpack_from("<4H", body, i) for i in range(0, min(len(body), 8*40), 8)]
for i, r in enumerate(recs):
    print(f"  {i:3d}  w0=0x{r[0]:04X} w1={r[1]:>6} w2={r[2]:>6} w3=0x{r[3]:04X}")
