"""Usa os 382 arquivos de nivel como restricao para determinar o layout.

Nada de ajustar um arquivo so. Uma hipotese e aceita somente se for verdadeira
para TODOS os arquivos do corpus.
"""
import struct, sys, collections, os
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
import json
_manifest = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_manifest.json"),
                           encoding="utf-8"))
levels = sorted(r["path"] for r in _manifest
                if r["path"].startswith("course/") and r["path"].endswith(".bin")
                and "_bgdat" not in r["path"])
print(f"{len(levels)} arquivos de nivel (sem bgdat)\n")

def sections(blob):
    hdr = struct.unpack_from("<I", blob, 0)[0]
    pairs = [struct.unpack_from("<II", blob, off) for off in range(8, hdr - 7, 8)]
    return hdr, pairs

hdr_sizes = collections.Counter()
npairs = collections.Counter()
bad = []
sec_counts = []
for name in levels:
    blob = rom.getFileByName(name)
    hdr, pairs = sections(blob)
    hdr_sizes[hdr] += 1
    npairs[len(pairs)] += 1
    # valida: pares (offset,length) devem ladrilhar o arquivo sem buracos nem sobreposicao
    offs = [o for o, l in pairs if l]
    ends = [o + l for o, l in pairs if l]
    contiguous = sorted(offs) == sorted([pairs[0][0]] + ends[:-1]) if len(offs) > 1 else True
    tail_ok = max(ends) <= len(blob) and (len(blob) - max(ends)) < 16
    if not tail_ok:
        bad.append((name, len(blob), max(ends)))
    sec_counts.append(len([1 for o, l in pairs if l]))

print("tamanhos de cabecalho:", dict(hdr_sizes))
print("qtd de pares (offset,length):", dict(npairs))
print("secoes nao-vazias por arquivo:", dict(collections.Counter(sec_counts)))
print(f"arquivos cujo ultimo segmento nao fecha o arquivo: {len(bad)}")
for b in bad[:5]: print("   ", b)

# para cada indice de secao, quais strides dividem TODAS as ocorrencias?
per_index = collections.defaultdict(list)
for name in levels:
    blob = rom.getFileByName(name)
    hdr, pairs = sections(blob)
    for i, (o, l) in enumerate(pairs):
        per_index[i].append((name, o, l))

print("\n== por indice de secao: tamanho minimo/maximo e strides universais ==")
for i in sorted(per_index):
    lens = [l for _, _, l in per_index[i] if l]
    if not lens:
        print(f"  sec {i:>2}: sempre vazia ({len(per_index[i])} ocorrencias)"); continue
    strides = [s for s in (4, 6, 8, 10, 12, 16, 20, 24, 32) if all(l % s == 0 for l in lens)]
    print(f"  sec {i:>2}: n={len(lens):>3} tam min={min(lens):>6} max={max(lens):>7} "
          f"unicos={len(set(lens)):>3} strides que dividem todos: {strides}")
