"""Teste de sentinela: caminha registros de N bytes ate um registro-terminador e
verifica se o terminador cai exatamente na borda final da secao.

Somente o stride correto satisfaz isso em todos os arquivos do corpus.
"""
import struct, os, json
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
man = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_manifest.json"), encoding="utf-8"))
levels = sorted(r["path"] for r in man if r["path"].startswith("course/")
                and r["path"].endswith(".bin") and "_bgdat" not in r["path"])

def secs_of(blob):
    hdr = struct.unpack_from("<I", blob, 0)[0]
    base = struct.unpack_from("<I", blob, 4)[0]
    pairs = [struct.unpack_from("<II", blob, off) for off in range(8, hdr - 7, 8)]
    return [(base, pairs[0][0] - base)] + pairs

corpus = [(n, rom.getFileByName(n)) for n in levels]

TERMS = {
    "u16 0xFFFF na 1a palavra": lambda r: r[0] == 0xFFFF,
    "todos os u16 zero":        lambda r: all(v == 0 for v in r),
    "1a palavra 0xFFFF e resto 0": lambda r: r[0] == 0xFFFF and all(v == 0 for v in r[1:]),
}

for stride in (8, 12, 16, 20, 24):
    for tname, term in TERMS.items():
        ok = tot = 0
        fails = []
        for name, blob in corpus:
            s = secs_of(blob)
            if len(s) < 7: continue
            o, l = s[6]
            if l == 0: continue
            tot += 1
            k = 0
            while o + (k+1)*stride <= o + l:
                r = struct.unpack_from(f"<{stride//2}H", blob, o + k*stride)
                if term(r):
                    break
                k += 1
            else:
                fails.append((name, "sem terminador", l, k*stride))
                continue
            if o + k*stride + stride == o + l:
                ok += 1
            else:
                fails.append((name, f"para em {k*stride} de {l}", l, k))
        print(f"stride {stride:>2}  sentinela '{tname:<28}': fecha exatamente em {ok}/{tot}")
        if fails and ok/max(1,tot) > 0.8:
            print(f"   exemplos de falha: {fails[:3]}")
