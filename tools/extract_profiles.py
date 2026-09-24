"""Build the object-id -> actor-class table from the CC0 decompilation.

Why this route: NSMB resolves a placed object with `CurrentProfileTable[object_id]
->constructor()` (ref/nsmb-decomp/src/Bases/Base.cpp:185), and there are no name
strings in the ARM9 -- the game's ids are numeric. But the decompilation carries,
for each actor class, a header comment naming its MainProfileTable slot and a
definition whose second field is that same slot:

    // MainProfileTable slot 254  |  ov000  |  profile @ 0x020da9a8
    ActorProfile Object254_Profile = { SpinBlock::create, 254, 91, NULL };

Pairing those two gives (id -> class, overlay, profile address) without guessing.
Coverage is partial by nature: the decomp is early-stage and some classes (Goomba,
Koopa, Piranha Plant) have no profile line at all. Missing ids are reported as
missing rather than filled in from memory or from external tables.

The two fields are cross-checked: when the comment's slot and the struct's second
field disagree, the row is flagged instead of silently preferred.
"""
import json
import os
import re
import sys

DECOMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "nsmb-decomp", "src")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "nsmb_object_ids.json")

# "// MainProfileTable slot 254  |  ov000  |  profile @ 0x020da9a8"
SLOT_RE = re.compile(
    r"MainProfileTable slots? ([0-9, ]+?)\s*\|\s*ov([0-9]+)\s*\|\s*profile @ (0x[0-9a-fA-F]+)"
)
# "ActorProfile Object254_Profile = { SpinBlock::create, 254, 91, NULL };"
PROF_RE = re.compile(
    r"(\w+Profile)\s*=\s*\{\s*([\w:]+)::(create\w*)\s*,\s*([0-9xXa-fA-F]+)\s*,\s*([0-9xXa-fA-F]+)\s*(?:,\s*([^;]*))?\}",
    re.S,
)


def files():
    for root, _, names in os.walk(DECOMP):
        for n in names:
            if n.endswith((".hpp", ".cpp")):
                yield os.path.join(root, n)


def main():
    # A profile comment sits above the class declaration in a header, while the
    # Profile struct lives in the .cpp. So collect slots per *class stem* (the
    # filename without extension) and join on that.
    slots = {}
    for path in files():
        stem = os.path.splitext(os.path.basename(path))[0]
        text = open(path, encoding="utf-8", errors="replace").read()
        for m in SLOT_RE.finditer(text):
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            for n in nums:
                slots.setdefault(stem, []).append(
                    {"id": n, "overlay": int(m.group(2)), "profile_addr": m.group(3),
                     "src": os.path.relpath(path, DECOMP)}
                )

    rows, conflicts, seen = [], [], set()
    for path in files():
        text = open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, DECOMP)
        for m in PROF_RE.finditer(text):
            sym, cls, fn, f2, f3, rest = m.groups()
            oid = int(f2, 0)
            render = int(f3, 0)
            loader = None
            if rest and "NULL" not in rest:
                lm = re.search(r"/\*\s*(TODO:\s*)?(0x[0-9a-fA-F]+)", rest)
                loader = lm.group(2) if lm else rest.strip()
            stem = os.path.splitext(rel)[0].split(os.sep)[-1]
            cmt = slots.get(stem) or slots.get(cls)
            agree = None
            if cmt:
                agree = any(c["id"] == oid for c in cmt)
                if not agree:
                    conflicts.append({"class": cls, "struct_id": oid,
                                      "comment_ids": [c["id"] for c in cmt], "src": rel})
            key = (oid, cls)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": oid, "class": cls, "create": fn, "render_priority": render,
                         "resource_loader": loader, "slot_comment_agrees": agree, "src": rel})

    rows.sort(key=lambda r: r["id"])
    ids = {}
    for r in rows:
        ids.setdefault(str(r["id"]), []).append(r["class"])

    out = {
        "source": "ref/nsmb-decomp (CC0 clean-room decompilation, early stage)",
        "lookup": "CurrentProfileTable[object_id]->constructor(); see src/Bases/Base.cpp:185",
        "units": {"fixed_point": "0x1000 == 1 pixel (12 fractional bits)",
                  "evidence": "src/AAA.hpp:17 _FixedFlt(flt)*4096; src/Bases/StageEntity.cpp:100 "
                              "viewOffset.x << 0xc; src/Bases/Coin.cpp:650 vs :705 (8 px == 0x8000)"},
        "tile_px": 16,
        "fps": 60,
        "profiles": rows,
        "id_to_classes": ids,
        "slot_comment_conflicts": conflicts,
        "max_id": max((r["id"] for r in rows), default=0),
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"{len(rows)} profile rows, {len(ids)} distinct ids, max id {out['max_id']}")
    print(f"slot-comment cross-check: {sum(1 for r in rows if r['slot_comment_agrees'])} agree, "
          f"{sum(1 for r in rows if r['slot_comment_agrees'] is False)} disagree, "
          f"{sum(1 for r in rows if r['slot_comment_agrees'] is None)} with no comment")
    dup = {k: v for k, v in ids.items() if len(v) > 1}
    print(f"ids claimed by more than one class: {len(dup)}", list(dup.items())[:5])
    print(f"saved to {os.path.relpath(OUT, os.getcwd())}")


if __name__ == "__main__":
    sys.exit(main())
