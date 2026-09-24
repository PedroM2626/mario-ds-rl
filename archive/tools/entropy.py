import math, struct, sys
data = open(sys.argv[1], "rb").read()
def ent(b):
    if not b: return 0.0
    h=[0]*256
    for x in b: h[x]+=1
    n=len(b)
    return -sum((c/n)*math.log2(c/n) for c in h if c)
print("offset      entropy  zeros  printable-ascii")
step = 1<<20
for off in range(0, len(data), step):
    b = data[off:off+step]
    z = b.count(0)/len(b)
    printable = sum(1 for x in b if 32 <= x < 127)/len(b)
    print(f"0x{off:08X}  {ent(b):.3f}   {z*100:5.1f}%  {printable*100:5.1f}%")
print()
print("tail 0x200:", data[-0x40:].hex())
n=data.count(b"NCCL"); print("NCCL count:", n)
for m in (b"NDFA",b"FAT0",b"FREF",b"NAM0",b"NRPO",b"OBAR",b"VMD ",b"SMK ",b"SMD "):
    print(m, data.find(m))
