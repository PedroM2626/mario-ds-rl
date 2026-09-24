"""Emit the training-ready dataset: one JSON line per level, plus a physics sidecar.

Design decisions that matter for a consumer:
  * No emulator is required to read this. Everything comes from the ROM plus the
    CC0 decompilation.
  * The record is (id, x_tiles, y_tiles, w3, settings, w5) at a 12-byte stride from
    offset 0, ended by a 0xFFFFFFFF u32. That framing is proven by the terminator
    appearing in 189/189 levels with a non-empty list, and cross-checked by y
    agreeing with an independent 20-byte reading of the same bytes taken earlier.
    Words 3 and 5 are emitted raw because their roles are not known.
  * x_px / y_px are the tile fields multiplied by 16, so they are tile-centre-ish
    pixels, not sub-tile positions. Physics constants in nsmb_physics.json are in
    0x1000 == 1 px, a different scale; convert before mixing them.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LEVELS = os.path.join(ROOT, "out", "nsmb_levels.json")
PHYS = os.path.join(ROOT, "out", "nsmb_physics.json")
OUT = os.path.join(ROOT, "out", "dataset")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(LEVELS))
    phys = json.load(open(PHYS))
    known = {int(k): v for k, v in
             json.load(open(os.path.join(ROOT, "out", "nsmb_object_ids.json")))
             ["id_to_classes"].items()}

    n_ent = 0
    with open(os.path.join(OUT, "levels.jsonl"), "w", encoding="utf-8") as fh:
        for name, lv in sorted(d["levels"].items()):
            ents = []
            for r in lv["objects"]:
                ents.append({
                    "id": r["u16"][0], "class": r.get("id_class"),
                    "x_px": r["x_px"], "y_px": r["y_px"],
                    "u16_3": r["u16"][3], "settings": r["u16"][4], "u16_5": r["u16"][5],
                })
            n_ent += len(ents)
            fh.write(json.dumps({
                "level": name,
                "course_id": name.split("/")[-1].split(".")[0],
                "file_size": lv["file_size"],
                "objects": ents,
                "sections": [{"index": s["index"], "offset": s["offset"],
                              "length": s["length"], "stride": s["stride"]}
                             for s in lv["sections"]],
            }, ensure_ascii=False) + "\n")

    # a flat table is what most trainers want first: one row per placed entity
    with open(os.path.join(OUT, "entities.csv"), "w", encoding="utf-8") as fh:
        fh.write("level,course_id,id,class,x_px,y_px,u16_3,settings,u16_5\n")
        for name, lv in sorted(d["levels"].items()):
            cid = name.split("/")[-1].split(".")[0]
            for r in lv["objects"]:
                u = r["u16"]
                # a class list would break the CSV; there is at most one class per id
                cls = (r.get("id_class") or [""])[0].replace(",", ";")
                fh.write(f"{name},{cid},{u[0]},{cls},{r['x_px']},{r['y_px']},"
                         f"{u[3]},{u[4]},{u[5]}\n")

    meta = {
        "schema_version": 1,
        "levels": len(d["levels"]), "entities": n_ent,
        "units": phys["units"],
        "id_column": "col0 of the 12-byte record; framing proven by the 0xFFFFFFFF terminator u32 in 189/189 non-empty levels",
        "coordinates": "x_px/y_px are record words 1 and 2 times 16 (tiles -> pixels); z/settings words 3..5 are raw",
        "readme": "levels.jsonl = one level per line (entities + section inventory); "
                  "entities.csv = flat table; physics.json = engine constants and the "
                  "integration model; object_ids.json = 342 object-id -> actor class.",
        "what_is_not_here": [
            "the player's own gravity and jump velocity: not present in the decompilation",
            "terrain tile layout: the level files store chunk references; the chunk "
            "library is BG_chk/*MainUnitChangeData.bin and is not decoded yet",
            "the meaning of words 3..5 of the record (z? generator settings?)",
        ],
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    import shutil
    shutil.copyfile(PHYS, os.path.join(OUT, "physics.json"))
    shutil.copyfile(os.path.join(ROOT, "out", "nsmb_object_ids.json"),
                    os.path.join(OUT, "object_ids.json"))

    for f in sorted(os.listdir(OUT)):
        print(f"  {f:<18} {os.path.getsize(os.path.join(OUT, f)) / 1e6:>7.2f} MB")
    print(f"{meta['levels']} levels, {meta['entities']} entities")


if __name__ == "__main__":
    main()
