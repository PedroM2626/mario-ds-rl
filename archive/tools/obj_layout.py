"""Prova o layout da secao de objetos e nomeia os campos por estatistica de corpus.

Teste 1: se o primeiro u32 de uma secao for igual a (len-4)/stride para todos os
arquivos, a hipotese "contador + registros" esta confirmada.
Teste 2: identificar qual campo e ID (baixa cardinalidade no corpus inteiro),
qual e X (monotonicamente crescente dentro da lista) e qual e Y (limitado).
"""
import struct, os, json, collections
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

print("== teste do contador: primeiro u32 == (len-4)/stride ? ==")
cands = []
for i in range(14):
    hits = tot = 0
    for name, blob in corpus:
        s = secs_of(blob)
        if i >= len(s): continue
        o, l = s[i]
        if l < 16 or l % 4: continue
        tot += 1
        cnt = struct.unpack_from("<I", blob, o)[0]
        for stride in (8, 12, 16, 20, 24):
            if (l - 4) % stride == 0 and cnt == (l - 4) // stride:
                hits += 1; cands.append((i, stride)); break
    if tot:
        print(f"  sec {i:>2}: bate em {hits}/{tot} arquivos")

best = collections.Counter(cands).most_common(5)
print(f"\ncombos (secao, stride) mais consistentes: {best}\n")

if not best:
    raise SystemExit("nenhum contador confirmado")

sec_i, stride = best[0][0]
nfields = stride // 2
print(f"== analisando sec {sec_i} como u32 contador + registros de {stride}B ({nfields} x u16) ==")
per_level = []
allvals = [[] for _ in range(nfields)]
for name, blob in corpus:
    s = secs_of(blob)
    if sec_i >= len(s): continue
    o, l = s[sec_i]
    if l < 8: continue
    cnt = struct.unpack_from("<I", blob, o)[0]
    recs = [struct.unpack_from(f"<{nfields}H", blob, o + 4 + k*stride) for k in range(cnt)]
    if not recs: continue
    per_level.append((name, recs))
    for r in recs:
        for f in range(nfields):
            allvals[f].append(r[f])

for f in range(nfields):
    v = allvals[f]
    distinct = len(set(v))
    mono = sum(sum(1 for a, b in zip([r[f] for r in rs], [r[f] for r in rs][1:]) if b >= a)
               for _, rs in per_level)
    pairs_tot = sum(len(rs) - 1 for _, rs in per_level if len(rs) > 1)
    frac = mono / pairs_tot if pairs_tot else 0
    print(f"  campo {f}: min={min(v):>6} max={max(v):>6} distintos={distinct:>5} "
          f"monotono-nao-decrescente em {frac:6.1%} dos pares")

print("\n== amostra: 12 primeiros registros de course/A01_1.bin ==")
for name, recs in per_level:
    if name.endswith("A01_1.bin"):
        for i, r in enumerate(recs[:12]):
            print(f"  {i:>3}  " + "  ".join(f"f{j}=0x{v:04X}({v})" for j, v in enumerate(r)))
        break
json.dump({"sec": sec_i, "stride": stride}, open("obj_layout.json", "w"))
