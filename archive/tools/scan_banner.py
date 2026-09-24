import struct, sys, pathlib

p = sys.argv[1]
data = open(p, "rb").read()
banner = struct.unpack_from("<I", data, 0x40)[0]
rom_end = struct.unpack_from("<I", data, 0x48)[0]
print(f"banner offset 0x{banner:X}, rom end 0x{rom_end:X}, banner len {rom_end-banner}")
blk = data[banner:banner+0x2400]
print("banner u32[0..5]:", [hex(struct.unpack_from("<I", blk, o)[0]) for o in range(0, 0x18, 4)])
base = struct.unpack_from("<I", blk, 0x20)[0]
print(f"title block at +0x20 -> 0x{base:X}")
for i in range(5):
    off = base + i*0x100
    raw = blk[off:off+0x100]
    try:
        s = raw.decode("utf-16-le").split("\x00")[0]
    except Exception as e:
        s = f"<{e}>"
    print(f"  lang{i}: {s!r}")

# filesystem table: 5th u32 of header area at 0x4C is header-ext offset
ext = struct.unpack_from("<I", data, 0x4C)[0]
print(f"\nheader ext offset 0x{ext:X}")
if ext and ext < len(data):
    print("ext bytes:", data[ext:ext+0x20].hex())

# locate binary FS: find "YSARC" / look at fixed table copy usually at 0x40 in ROM area
for magic in (b"YSARC", b"NDSN", b"\x2a\x2a\x4e\x41", b"NCCL", b"NSCR", b"SCET"):
    i = data.find(magic)
    print(f"{magic!r}: first at {hex(i) if i>=0 else 'nao encontrado'}")
