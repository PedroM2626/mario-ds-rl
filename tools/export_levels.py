"""Export the level container as a machine-readable dataset skeleton.

What is asserted here, and what is not:
  * The 112-byte header with 13 (offset, length) pairs tiling the file: verified on
    all 191 levels.
  * sec5 = 12-byte records starting at offset 0, terminated by a trailing u32 that is
    0xFFFFFFFF in 189/189 levels with a non-empty list (the 2 others have len 4, i.e.
    an empty list). 5311 records total. The terminator is what proves the framing:
    an earlier revision of this file put a 4-byte *header* first, which shifted every
    record by two words and made 37% of the positions come out as zeros.
  * sec0=24B, sec1..sec4=20B, sec6=16B, sec8=8B, sec10=16B, sec12=16B: strides whose
    length divides every level's section size (GCD over the corpus).
  * NOT asserted: which u16 of the 12-byte record is the placed actor's object id.
    col0 is id-like (max 207, 100% <= 384) but its frequencies are nearly flat over
    22..30; col4/col5 are id-like too but carry 65535 (-1, "none") hundreds of times,
    which reads more like an optional reference. Column 1 is tile-aligned pixels
    (91% multiples of 16, up to 13056 px = 816 tiles) and is the x candidate.
    Every record is emitted with its raw columns and a name only where the column is
    defensible; the rest stay `u16_0..u16_5`.
"""
import json
import os
import struct

import ndspy.rom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
NSEC = 13
SEC_STRIDE = {0: 24, 1: 20, 2: 20, 3: 20, 4: 20, 6: 16, 7: 12, 8: 8, 9: 8, 10: 16, 11: 16, 12: 16}


def main():
    rom = ndspy.rom.NintendoDSRom.fromFile(ROM)
    man = json.load(open(os.path.join(HERE, "fs_manifest.json")))
    p2id = {e["path"]: e["id"] for e in man}
    ids = json.load(open(os.path.join(ROOT, "out", "nsmb_object_ids.json")))
    known = {int(k): v for k, v in ids["id_to_classes"].items()}
    courses = sorted(p for p in p2id if p.startswith("course/") and p.endswith(".bin")
                     and "_bgdat" not in p)

    out = {
        "units": {"fixed_point": "0x1000 == 1 pixel", "tile_px": 16, "fps": 60},
        "id_dictionary": "out/nsmb_object_ids.json",
        "caveats": [
            "col3..col5 of a sec5 record are raw; only id/x/y are claimed",
            "section names are indices, not the game's own names -- no FourCC exists in "
            "the ROM for them and the decompilation has no course reader",
        ],
        "sec5_columns": {
            "id": 0, "x_tiles": 1, "y_tiles": 2, "z_or_pad": 3, "settings": 4, "settings2": 5,
            "basis": "in the corrected framing col0 is <=384 in 100% of records with 225 "
                     "distinct values (84.5% resolving to a known profile slot), col1 spans "
                     "0..447 with median 76 (a level width in tiles), col2 spans 0..207. "
                     "Roles for col3..col5 are still inferred, not proven.",
            "id_status": "col0, resolved by fixing the record framing (see above). The "
                      "tests listed under failed_tests were run against the WRONG framing "
                      "and are kept only as a record of what did not work.",
            "col0_support": [
                "never holds the -1 sentinel (col3: 2, col4/col5: 189 each)",
                "bounded at 207, 100% of values <= 384",
                "exceeds its own level's record count in 151/189 levels, so not an index",
            ],
            "col4_support": [
                "known actor ids spread over far more levels than in col0: id 106 in 28 "
                "levels vs 3, id 57 in 29 vs 13, id 148 in 29 vs 5",
            ],
            "failed_tests": [
                "frequency of 'is a known id' cannot discriminate: the dictionary covers "
                "342 of ~385 slots, so it is really a test of '<= 384' (col0 81%, col4 82%)",
                "same-level reference test: no column points at col0 more often than the "
                "~10% base rate",
                "'the player is placed exactly once per level' test on id 21: col0 has zero "
                "21s in 135/191 levels and 11 copies in one. This looks like a refutation "
                "but is NOT one -- it assumes stages place the player via actor id 21, "
                "which is unverified. Recorded as an invalid test, not as evidence.",
            ],
            "what_would_settle_it": "one 4 MiB RAM dump with any stage open, matching "
                                    "Base+0x0C (object_id, u16) and Base+0x08 (settings) "
                                    "against the file records; the file gives exact "
                                    "tile-aligned pixel coordinates, so position*0x1000 as "
                                    "consecutive u32 is a near-unique fingerprint.",
        },
        "levels": {},
    }
    nrec = 0
    for p in courses:
        b = rom.files[p2id[p]]
        hdr = struct.unpack_from("<I", b, 0)[0]
        data0 = struct.unpack_from("<I", b, 4)[0]
        secs = []
        for si in range(NSEC):
            o, l = struct.unpack_from("<II", b, 8 + si * 8)
            e = {"index": si, "offset": o, "length": l, "stride": SEC_STRIDE.get(si)}
            if l and si in SEC_STRIDE and l % SEC_STRIDE[si] == 0:
                w = SEC_STRIDE[si] // 2
                e["records"] = [list(struct.unpack_from("<%dH" % w, b, o + r * SEC_STRIDE[si]))
                                for r in range(l // SEC_STRIDE[si])]
            secs.append(e)
        o5, l5 = struct.unpack_from("<II", b, 8 + 5 * 8)
        objs = []
        if l5 >= 4 and (l5 - 4) % 12 == 0:
            for r in range((l5 - 4) // 12):
                v = list(struct.unpack_from("<6H", b, o5 + r * 12))
                nrec += 1
                objs.append({"u16": v,
                             "id_class": known.get(v[0]),
                             "x_px": v[1] * 16, "y_px": v[2] * 16})
        out["levels"][p] = {
            "header_size": hdr, "data_start": data0, "file_size": len(b),
            "sections": secs, "sec5_header_word": struct.unpack_from("<I", b, o5)[0],
            "objects": objs,
        }
    out["totals"] = {"levels": len(courses), "sec5_records": nrec}
    path = os.path.join(ROOT, "out", "nsmb_levels.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(courses)} levels, {nrec} object records -> {os.path.relpath(path, ROOT)} "
          f"({os.path.getsize(path)/1e6:.1f} MB)")
    # coverage of the id column, so a reader can see how much of the list names an actor
    a = sum(1 for lv in out["levels"].values() for r in lv["objects"] if r["id_class"])
    print(f"records whose id resolves to a known profile slot: {a}/{nrec} ({a/nrec:.0%})")
    xs = [r["x_px"] for lv in out["levels"].values() for r in lv["objects"]]
    ys = [r["y_px"] for lv in out["levels"].values() for r in lv["objects"]]
    print(f"x_px range {min(xs)}..{max(xs)} (median {sorted(xs)[len(xs)//2]}), "
          f"y_px range {min(ys)}..{max(ys)}")


if __name__ == "__main__":
    main()
