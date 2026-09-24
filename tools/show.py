import json, sys, collections
rows = json.load(open("fs_manifest.json", encoding="utf-8"))

def clean(m):
    if all(32 <= ord(c) < 127 for c in m):
        return m
    return m.encode("latin-1", "ignore").hex()

by_magic = collections.Counter(clean(r["magic"]) for r in rows)

def show(prefix, n=30):
    sel = [r for r in rows if r["path"].startswith(prefix)]
    total = sum(r["size"] for r in sel)
    print(f"--- {prefix}  ({len(sel)} arquivos, {total:,} bytes) ---")
    for r in sel[:n]:
        print(f"  #{r['id']:<5} {r['size']:>8,}  {clean(r['magic']):<8} {r['path']}")
    if len(sel) > n:
        print(f"  ... +{len(sel)-n} outros")
    print()

for p in sys.argv[1:] or ["course/", "obj/", "player/", "script/", "BG_chk/", "BG_nsc/", "ARCHIVE/", "map/"]:
    show(p)
