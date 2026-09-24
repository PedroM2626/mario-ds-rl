import struct, sys

path = sys.argv[1]
data = open(path, "rb").read()
size = len(data)

def u32(o): return struct.unpack_from("<I", data, o)[0]
def u16(o): return struct.unpack_from("<H", data, o)[0]

title = data[0x00:0x0C].rstrip(b"\x00")
code = data[0x0C:0x10].decode("ascii", "replace")
maker = data[0x10:0x12].decode("ascii", "replace")
unit = data[0x13]
cap = 16 << data[0x14] if data[0x14] < 20 else data[0x14]

print(f"file size      : {size:,} bytes ({size/1024/1024:.1f} MiB)")
print(f"title          : {title!r}")
print(f"game code      : {code}   maker: {maker}   unitcode: 0x{unit:02X}")
print(f"declared capacity: {cap:,} KiB")
print()

fields = [
    ("ARM9 offset", 0x20), ("ARM9 entry", 0x24), ("ARM9 load", 0x28), ("ARM9 size", 0x2C),
    ("ARM7 offset", 0x30), ("ARM7 entry", 0x34), ("ARM7 load", 0x38), ("ARM7 size", 0x3C),
    ("icon/banner", 0x40), ("logo/secure area", 0x44), ("ROM end", 0x48),
    ("header ext", 0x4C), ("ARM9 iover", 0x64), ("ARM7 iover", 0x68),
    ("ROM region", 0x6C), ("RAM region", 0x70),
]
for name, off in fields:
    v = u32(off)
    flag = "" if v == 0 else ("" if v < size else "  <-- fora do arquivo: cabecaprovavelmente ENCRIPTADA (KEY1)")
    print(f"{name:<16} 0x{off:03X} = 0x{v:08X} ({v:>10,}){flag}")

print()
# KEY1 "encrypted overlays" table lives at 0x180; banner at icon offset
print("bytes @0x180 (KEY1 block):", data[0x180:0x190].hex())
enc_flag = u32(0x180) & 0xFF
print(f"overlay-encryption indicator: 0x{enc_flag:02X}")
banner = u32(0x40)
if 0 < banner < size:
    print(f"banner magic @0x{banner:X}: {data[banner:banner+4].hex()} (0x340 = DS banner)")
# entropy sanity on a mid-file chunk: high entropy + garbage offsets => encrypted
import math
def ent(b):
    h = [0]*256
    for x in b: h[x]+=1
    return -sum((c/len(b))*math.log2(c/len(b)) for c in h if c)
chunk = data[size//2:size//2+65536]
print(f"mid-file entropy: {ent(chunk):.3f} bits/byte")
