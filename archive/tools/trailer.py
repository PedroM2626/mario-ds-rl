import sys
data = open(sys.argv[1], "rb").read()
n = len(data)
# find true end of non-0xFF / non-0x00 padding
i = n - 1
while i > 0 and data[i] in (0xFF, 0x00):
    i -= 1
print(f"file size 0x{n:X}; last non-padding byte at 0x{i:X} (value 0x{data[i]:02X})")
print("bytes around end of data:", data[i-0x20:i+0x20].hex())
print("last 16 bytes        :", data[-16:].hex())
print()
# KEY1 trailer (GBATEK 10.3): pointer block usually sits near end of the card
for probe in (i+1, i+5, n-8, n-4):
    if 0 <= probe < n:
        print(f"u32 @0x{probe:X} = {int.from_bytes(data[probe:probe+4],'little'):,}")
print()
print("head of data area @0x4000 :", data[0x4000:0x4040].hex())
print("head @0x6000             :", data[0x6000:0x6040].hex())
print("head @0x100000           :", data[0x100000:0x100040].hex())
print("bytes @0x260418 (hdr 'ROM end'):", data[0x260418:0x260458].hex())
# longest run of 0x20+ spaces / ascii words -> code rodata vs ciphertext
words = data[0x4000:i].split(b"\x00")
good = [w for w in words if len(w) > 6 and all(32 <= c < 127 for c in w)]
print(f"\nASCII strings >6ch in data area: {len(good)} (exemplos: {good[:6]})")
