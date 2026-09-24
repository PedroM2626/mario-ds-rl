"""Teste decisivo: KEY1-encrypted vs decrypted cleanrip."""
import struct, sys, collections

data = open(sys.argv[1], "rb").read()

# 1) KEY1 produz keystream por palavra de 4 bytes, ciclando um array KeyX de 0x104
#    bytes (=65 palavras). Ciphertext real NAO tem periodo 4 longo em regioes de
#    plaintext repetitivo; plaintext repetitivo nao-criptografado tem.
region = data[0x4000:0x4200]
w = [region[i:i+4] for i in range(0, len(region), 4)]
cnt = collections.Counter(w)
top, freq = cnt.most_common(1)[0]
print(f"[1] bloco @0x4000: {len(w)} palavras de 4B, mais frequente {top.hex()} aparece {freq}x ({freq/len(w)*100:.1f}%)")
print(f"    palavras unicas: {len(cnt)}/{len(w)}")

# 2) corrida longa de byte identico: fluxo de cifra destroi isto
def longest_run(b):
    best=(0,0,0); cur=1
    for i in range(1,len(b)):
        if b[i]==b[i-1]: cur+=1
        else:
            if cur>best[0]: best=(cur,b[i-1],i-cur)
            cur=1
    if cur>best[0]: best=(cur,b[-1],len(b)-cur)
    return best
for name,off,ln in [("0x4000-0x260418",0x4000,0x25C418),("area media 0x800000",0x800000,0x100000)]:
    run,val,where = longest_run(data[off:off+ln])
    print(f"[2] {name}: maior corrida = {run} x 0x{val:02X} em 0x{off+where:X}")

# 3) magics de arquivos Nintendo / strings de engine
for m in [b"NCCL",b"NSCR",b"NSBC",b"NDRM",b"NDS ",b"NRPO",b"NTR",b"LZ",b"11",b".bin",b".arc",b"arc",
          b"nitro",b"Nitro",b"SCULPT",b"ROOT",b"FAT",b"NAM0",b"FREF",b"GAME",b"GAME_OVER"]:
    i = data.find(m)
    if i >= 0: print(f"[3] magic {m!r} em 0x{i:X}")

# 4) ndspy faz o parse autoritativo
print("\n[4] ndspy:")
try:
    import ndspy.rocket, ndspy.nitro, ndspy.zigzag
    rom = None
    try:
        from ndspy.nitro import Rom as NitroRom
    except Exception as e:
        print("    import falhou:", e)
except Exception as e:
    print("    ndspy modules:", e)

try:
    import ndspy.ctr
except Exception as e:
    pass

try:
    from ndspy.rocket import Rom as RocketRom
    r = RocketRom(data)
    print("    ROM DSi/Rocket parseado:", len(r.fileManager.files), "arquivos")
except Exception as e:
    print(f"    rocket: {type(e).__name__}: {e}")

try:
    from ndspy.nitro import Rom as NR
    n = NR(data)
    print("    nitro parseado! arquivos:", len(n.fileManager.files))
except Exception as e:
    print(f"    nitro: {type(e).__name__}: {e}")

# 5) campo de flags de sobreposicao criptografada (GBATEK 0x180..0x198) e trailer
print(f"\n[5] header 0x180..0x1A0: {data[0x180:0x1A0].hex()}")
print(f"    u32@0x180(overlay flag)=0x{struct.unpack_from('<I',data,0x180)[0]:X}")
print(f"    u32@0x184={struct.unpack_from('<I',data,0x184)[0]:,}  u32@0x188={struct.unpack_from('<I',data,0x188)[0]:,}")
