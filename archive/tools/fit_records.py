"""Ajusta automaticamente o layout dos registros de objeto testando hipoteses.

Para cada secao-candidata, tenta varios (deslocamento, formato de campos) e
pontua pela plausibilidade: X monotonicamente crescente, X/Y dentro de faixa
de nivel, IDs pequenos e poucos valores 0xFFFF.
"""
import struct, sys
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())

FORMATS = {
    "3x u16 / 4x u16 (8B)":  ("<4H", 8),
    "u32 x + u32 y + u16 + u16 (12B)": ("<2I2H", 12),
    "u16 id+u16 p+u32 x+u32 y (12B)":  ("<2H2I", 12),
    "4x u32 (16B)":          ("<4I", 16),
    "8x u16 (16B)":          ("<8H", 16),
}

def sections(blob):
    hdr = struct.unpack_from("<I", blob, 0)[0]
    base = struct.unpack_from("<I", blob, 4)[0]
    out = [(base, 0)]
    pairs = []
    for off in range(8, hdr - 7, 8):
        pairs.append(struct.unpack_from("<II", blob, off))
    # primeiro bloco implicito: base ate inicio da primeira secao
    if pairs:
        out[0] = (base, pairs[0][0] - base)
    out += [(o, l) for o, l in pairs]
    return out

def score(recs, ix, iy, iid):
    if not recs: return -1, 0
    xs = [r[ix] for r in recs]; ys = [r[iy] for r in recs]; ids = [r[iid] for r in recs]
    if max(xs) == 0: return -1, 0
    mono = sum(1 for a, b in zip(xs, xs[1:]) if b >= a) / max(1, len(xs)-1)
    okx = sum(1 for v in xs if 0 <= v <= 40000) / len(xs)
    oky = sum(1 for v in ys if 0 <= v <= 3000) / len(ys)
    okid = sum(1 for v in ids if v < 0x400) / len(ids)
    noff = sum(1 for v in xs+ys+ids if v in (0xFFFF,)) / max(1, len(xs)*3)
    s = mono*0.35 + okx*0.25 + oky*0.25 + okid*0.15 - noff*0.6
    return s, mono

results = []
for lvl in sys.argv[1:] or ["course/A01_1.bin", "course/A02_1.bin", "course/A05_1.bin"]:
    try:
        blob = rom.getFileByName(lvl)
    except Exception as e:
        print(f"{lvl}: falha {e}"); continue
    print(f"\n===== {lvl} ({len(blob)} bytes) =====")
    for o, ln in sections(blob):
        if ln < 16 or ln % 8: continue
        best = None
        for fmt, (fstr, stride) in FORMATS.items():
            if stride < 8: continue
            n = ln // stride
            try:
                recs = [struct.unpack_from(fstr, blob, o + i*stride) for i in range(n)]
            except Exception:
                continue
            nf = len(recs[0])
            for iid in range(nf):
                for ix in range(nf):
                    for iy in range(nf):
                        if len({iid,ix,iy}) < 3: continue
                        s, mono = score(recs, ix, iy, iid)
                        if best is None or s > best[0]:
                            best = (s, fmt, stride, (iid,ix,iy), mono, n)
        if best and best[0] > 0.75:
            s, fmt, stride, pos, mono, n = best
            print(f"  secao 0x{o:05X} len {ln:>4} ({n:>3} x {stride}B)  score {s:.3f}  "
                  f"formato={fmt} campos(id,x,y)=idx{pos} monotonia X={mono:.0%}")
            results.append((lvl, o, ln, fmt, stride, pos))
print(f"\n{len(results)} secoes encaixaram com score > 0.75")
