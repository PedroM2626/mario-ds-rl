"""Extrai os dados de nivel do NSMB DS para JSON, com validacao pelo corpus.

LAYOUT APURADO (duas fontes independentes concordando):
  * cabecalho: u32 tamanho (0x70) + u32 inicio dos dados + 13 pares (offset, length)
    que ladrilham o arquivo inteiro.
  * secao 5: registros de 20 bytes  -> coincide com StageEntity::ObjectInfo da
    decomp CC0 (position,size,spawnOffset,viewOffset = 8x s16; + properties,
    + spawnSettings = u16) = 20 bytes exatos.
  * secao 6: u32 + N registros de 12 bytes (s16 com SINAL, nao u16).
"""
import struct, os, json, collections, sys
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(OUT, exist_ok=True)

rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
man = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_manifest.json"), encoding="utf-8"))
levels = sorted(r["path"] for r in man if r["path"].startswith("course/")
                and r["path"].endswith(".bin") and "_bgdat" not in r["path"])

def sections(blob):
    hdr = struct.unpack_from("<I", blob, 0)[0]
    base = struct.unpack_from("<I", blob, 4)[0]
    pairs = [struct.unpack_from("<II", blob, off) for off in range(8, hdr - 7, 8)]
    return [(base, pairs[0][0] - base)] + pairs

def decode_sec6(blob, o, l):
    """4 bytes de cabecalho + registros de 12 bytes lidos com s16 com sinal."""
    head = struct.unpack_from("<I", blob, o)[0]
    n = (l - 4) // 12
    recs = []
    for k in range(n):
        f = struct.unpack_from("<6h", blob, o + 4 + k*12)
        recs.append(f)
    return head, recs

stats = collections.Counter()
xs_all, ys_all, id_all = [], [], []
per_file = {}
width_hist = collections.Counter()

for name in levels:
    blob = rom.getFileByName(name)
    secs = sections(blob)
    o5, l5 = secs[5]; o6, l6 = secs[6]
    # secao 5: 20 bytes, StageEntity::ObjectInfo
    ents = []
    for k in range(l5 // 20):
        v = struct.unpack_from("<8h2H", blob, o5 + k*20)
        ents.append(v)
    head6, recs6 = decode_sec6(blob, o6, l6)
    per_file[name] = (ents, head6, recs6)
    for r in recs6:
        xs_all.append(r[1]); ys_all.append(r[2]); id_all.append(r[0])
    if recs6:
        stats["com_registros"] += 1
        # monotonia de X sobre s16
        seq = [r[1] for r in recs6]
        asc = sum(1 for a, b in zip(seq, seq[1:]) if b >= a)
        stats["pares"] += max(0, len(seq)-1)
        stats["pares_crescentes"] += asc
        width_hist[max(seq) - min(seq)] += 1

print(f"{len(levels)} niveis processados")
print(f"secao 6: cabecalho u32 assume {len(set(v[1] for v in per_file.values()))} valores "
      f"distintos no corpus (amostra: {sorted(set(v[1] for v in per_file.values()))[:6]})")

def rng(name, vals):
    if not vals: print(f"  {name}: vazio"); return
    print(f"  campo {name}: min={min(vals):>7} max={max(vals):>7} distintos={len(set(vals)):>4} "
          f"median={sorted(vals)[len(vals)//2]}")

print(f"\n== campo a campo da secao 6 ({len(xs_all)*3} amostras, s16 com sinal) ==")
for i, nm in enumerate("f0 f1 f2 f3 f4 f5".split()):
    rng(nm, [r[i] for _, _, rs in per_file.values() for r in rs])

seq_ok = stats["pares_crescentes"] / max(1, stats["pares"])
print(f"\nmonotonia: f1 (candidato a X) cresce em {seq_ok:.1%} dos {stats['pares']} pares consecutivos")
print(f"-> se for ~100%, f1 e a coordenada X e os objetos estao ordenados por X")

print("\n== amostra: 10 registros de course/A01_1.bin ==")
ents, head6, recs6 = per_file["course/A01_1.bin"]
print(f"  cabecalho da secao 6 = {head6}   registros = {len(recs6)}")
for i, r in enumerate(recs6[:10]):
    print(f"   {i:>3}  " + " ".join(f"f{j}={v:>6}" for j, v in enumerate(r)))

json.dump({n: {"sec5": [list(x) for x in e], "head6": h,
               "sec6": [list(x) for x in r]} for n, (e, h, r) in per_file.items()},
          open(os.path.join(OUT, "levels_raw.json"), "w"), ensure_ascii=False)
print(f"\nbruto salvo em out/levels_raw.json")
