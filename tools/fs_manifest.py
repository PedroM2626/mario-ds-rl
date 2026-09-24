"""Varre o filesystem Nitro do ROM e gera manifest (caminho, id, tamanho, magic)."""
import sys, os, json, struct, collections
import ndspy.rom

path = sys.argv[1]
out_dir = os.path.dirname(os.path.abspath(__file__))
rom = ndspy.rom.NintendoDSRom(data=open(path, "rb").read())

entries = []  # (caminho, fileID)

def walk(folder, prefix):
    """ndspy.fnt.Folder: .files e lista de nomes em ordem de ID a partir de
    .firstID; .folders e lista de (nome, Folder)."""
    for i, name in enumerate(folder.files):
        fid = folder.firstID + i
        p = f"{prefix}/{name}" if prefix else name
        entries.append((p, fid))
    for subname, sub in folder.folders:
        walk(sub, f"{prefix}/{subname}" if prefix else subname)

walk(rom.filenames, "")
print(f"entradas resolvidas por nome: {len(entries)}")

print(f"entradas resolvidas por nome: {len(entries)}")

# mapa id -> caminho (pode haver ids duplicados/ausentes)
id2name = {}
for name, fid in entries:
    id2name.setdefault(fid, name)

rows = []
magic_count = collections.Counter()
top_dirs = collections.Counter()
for fid, blob in enumerate(rom.files):
    magic = blob[:4]
    try:
        m = magic.decode("ascii")
    except Exception:
        m = magic.hex()
    magic_count[m] += 1
    name = id2name.get(fid, f"<sem nome #{fid}>")
    top_dirs[name.split("/")[0]] += 1
    rows.append({"id": fid, "path": name, "size": len(blob), "magic": m})

with open(os.path.join(out_dir, "fs_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

total = sum(r["size"] for r in rows)
print(f"\n{len(rows)} arquivos, {total:,} bytes ({total/1024/1024:.1f} MiB) de conteudo\n")
print("== pastas de nivel 1 (qtd de arquivos) ==")
for d, c in top_dirs.most_common(25):
    print(f"  {c:>5}  {d}")
print("\n== magic de 4 bytes mais comuns ==")
for m, c in magic_count.most_common(20):
    print(f"  {c:>5}  {m!r}")
print("\n== 25 maiores arquivos ==")
for r in sorted(rows, key=lambda r: -r["size"])[:25]:
    print(f"  {r['size']:>9,}  {r['magic']:<6} #{r['id']:<5} {r['path']}")
print("\nmanifest: tools/fs_manifest.json")
