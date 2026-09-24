# Superseded probes and falsified hypotheses

Everything in `tools/` and `bridge/` below this directory was run during the
reverse-engineering session on 2026-09-23 and later superseded. They are kept on
purpose: each one is a negative result, and the negative results are what
constrain the remaining search. Deleting them would let a future session repeat
the same mistakes.

## ROM structure (resolved, not superseded — kept in `tools/`)

The European ROM (`A2DP`, `0479 - New Super Mario Bros. (Europe)`) is **not**
encrypted. It parses cleanly with `ndspy` into 2449 named files across 191
levels, 368 enemy assets, 120 NARC archives.

Falsified along the way: an initial reading claimed the ROM was KEY1/`headcrypto`
encrypted because no `NCCL`/`NSCR`/`YSARC` magic was found. That was a wrong
premise — NSMB DS uses NARC/BMD0/BTX0/SDAT containers, not those. The claim was
killed by direct evidence: 5696-byte runs of `0x00` and plaintext strings inside
the ARM9 region, which a stream cipher cannot produce. No decryption code was
ever needed.

## Level file layout (partially resolved)

Established across all 191 levels. The reliable instrument here is the **GCD of
each section's length over every level**: a fixed-stride record array must have a
stride dividing 100% of the lengths, so the GCD reports the stride and the
divisors report the alternatives.

- Header is 112 bytes: `u32` header size, one standalone `u32` (0x20 in the
  samples), then 13 `(offset, length)` pairs that tile the file exactly to its
  last byte.
- Section strides that hold for **all** levels: `sec0` = 24 B, `sec1..sec4` = 20 B
  each, `sec6` = 16 B, `sec8` = 8 B (134 levels), `sec10` = 16 B, `sec12` = exactly
  16 B (one fixed struct, 189/191), `sec7` = 12 B (27 levels), `sec9`/`sec11` = 8 B
  and 16 B but present in only 17 levels.
- **`sec5` is the object stream** (4..1756 B). It is not a bare fixed-stride array --
  forcing one produced stride 16 for 58 levels and stride 8 for 45, which is
  self-contradictory -- because it starts with a 4-byte header word. See the next
  bullet for the structure that does hold.
- `sec4` (20 B records) decodes as geometry, not entities: column 0 and 1 are in
  [0, 5840] / [16, 3280] and overwhelmingly multiples of 16 (pixels / tile-aligned),
  with a small-int link index, a `0x1000 | n` field and a `0x8000`-flagged field.

Falsified:

- **"The 20-byte section matches `StageEntity::ObjectInfo`, so it is the placed-object
  list."** Wrong twice. The 20-byte arrays are sections 1-4, and in the decompilation
  `ObjectInfo` is *compile-time per-class metadata* with no object-id field, read by
  nobody except `Manhole::objectInfo` (`src/Bases/manhole.cpp:11`); `StageEntity.cpp`
  contains no file reading at all, and the class that would parse a course
  (`StageLayout`) is a `u8 _pad0[0xa8f4]` stub.
- **Any u16-column search for the object id.** Exhaustive over (section x stride x
  column) with a null baseline: best hit 65% against a 19.7% baseline, with maxima
  like 62465 where real object ids are <= 384. Note the trap in the check itself: the
  recovered id dictionary covers 342 of ~385 slots, so "is a known id" degenerates
  into "is <= 384" and cannot discriminate. Same for a bit-field version of the test.
- **Per-level monotonicity as evidence of a sorted key.** `sec4` column 8 is
  non-decreasing in 37/37 levels -- because it is constant (0 in 740 of 741 records).
  A monotonicity test must report variance, or it selects constants.
- `sentinel.py` — record lists are **not** terminated by a sentinel. Tested
  `0xFFFF`-first-word, all-zero, and `0xFFFF`+zeros terminators at strides
  8/12/16/20/24. All 0/191.
- `obj_layout.py` — sections do **not** carry a record-count header. Tested
  first-word-as-count across strides. 0/189.
- `fit_records.py`, `stride_fit.py`, `level_probe.py`, `course_sections.py` —
  blind field-width fitting. This is the methodological failure worth
  remembering: scoring 5 formats x all field permutations against a single file
  produced "score 1.0" on sections with 2 records. It is noise, not a decode.
  Replaced by `corpus_probe.py`, which requires a hypothesis to hold for all
  191 files simultaneously.

- **`sec5` is 12-byte records from offset 0, ended by a `0xFFFFFFFF` u32.** The
  terminator is present in 189/189 levels that have a non-empty list (the other 2
  have `len == 4`, i.e. an empty list), which is what actually proves the framing --
  `(len-4) % 12 == 0` alone cannot tell a 4-byte header from a 4-byte trailer, and an
  earlier revision of this file guessed "header" and shifted every record by two
  words, which is why 37% of positions came out as zeros. Record =
  `(id, x_tiles, y_tiles, w3, settings, w5)`; 5311 records, 85% of ids resolve to a
  known profile slot, x spans 0..447 tiles, y 0..207 tiles, and only 1% of entities
  sit at the origin.
  - Corrected same day: an earlier paragraph here claimed sec5 was "the only
    variable-length section, therefore bit-packed". The GCD of the raw lengths is 4
    because of the 4-byte header, not because of bit packing. Check `len - k` for
    small k before concluding anything about bitstreams.

Still open: which column is the placed actor's object id (col0 vs col4/col5), and
the meaning of the 4-byte header word.

## Player-coordinate search (failed; method abandoned)

Three candidates were found and then falsified:

| candidate | why it looked right | why it is wrong |
|---|---|---|
| `0x020A7678` | passed a freeze/move/freeze test with a plausible 2.90 px/frame walk | stayed bit-identical across 70 dumps covering 3 minutes of real play |
| `0x0209DFA4` (+`A8`, `AC`) | 60 Hz series showed 1.5 and 3.0 px/frame segments, matching walk and run | a later read gave a value 3 orders of magnitude smaller, and the neighbours are pointers into `0x0219xxxx` — it is a pointer table, not numeric fields |
| `guided2` 38-candidate set | clustered at regular 48-byte stride | that stride is an object pool, not the player |

Root cause, common to all three: searching 4 MiB for "something that moves like
a coordinate" using filters I had invented, with no independent anchor. Two
specific traps made it worse:

- `capture_walk.py` / `capture_stand.py` compared only **endpoints** of a window
  (A vs B). Any memory written once during the window passes. Fixed in
  `guided2.py` by requiring change *within* the walking window and bit-stability
  in both idle windows.
- Filters baked in the assumption that positions are 16.16 fixed point, so
  candidates in any other scale were rejected by construction.

`inject_scancode.py` and `capture_stand.py` also established that synthetic
key events do not reach the emulated DS in this setup — measured: walking changed
1310 words of RAM, standing still changed 1311. Human input is required for
labelled captures.

## Disassembly attempts (route A, blocked)

`find_gravity_code*.py`, `recursive_dis.py`, `find_player_ptr.py`,
`relocate_getplayer.py`.

The anchor was: the CC0 decompilation declares `Actor` fields in order
`velH, minVelH, accelV, minVelV, accelH` with `accelH` annotated at `0xC4`, so
`accelV` should sit at `0xBC`. Searching the European ARM9 for that offset:

- **Linear capstone disassembly produces ghosts.** It reported dozens of
  `LDC`/`STC`/`VSTR` instructions at `[rN,#0xbc]`. The DS ARM9 has no
  coprocessor; those were the disassembler running over literal pools and
  function boundaries. Never conclude "no such access exists" from a linear pass.
- **capstone stops at the first invalid byte** unless `md.skipdata = True`. With
  the default, a sweep of 465 KB returned 4 instructions and every derived count
  was a false zero.
- **The first `0x800` bytes of the ARM9 are `E7FFDEFF` fill**, not code — the
  entry point is `0x02000800`. Disassembling from offset 0 starts in filler.
- **A naive worklist is not enough.** One that follows only immediate `bl`/`b`
  reaches ~70 instructions and dies: the Nitro SDK boot stub hands off through
  an indirect branch plus an ARM/Thumb mode switch
  (`mov ip,#0x4000000; str ip,[ip,#0x208]; bl ...; mov r0,#0x13; msr cpsr_c,r0`).
- **Thumb has no 12-bit LDR displacement** on ARM946E-S (no Thumb-2): accessing
  `actor+0xBC` must be emitted as `ADD rX,#imm8` + `LDR rY,[rX,#disp]`, and the
  `ADD` immediate is in **bytes**, not scaled by 4. An earlier version scaled it
  and therefore found nothing.
- **Ghidra 12 headless specifics** (verified, cost three failed scripts):
  `Instruction.getReferences()` does not exist — use `CodeUnit.getReferencesFrom()`;
  `Reference.isDataReference()` does not exist — use `isMemoryReference()`;
  `Memory.getTotalLength()` and `AddressSpace.getLastAddress()` do not exist.
  Language IDs are `ARM:LE:32:v5t`, not the Ghidra-9 style `ARMLELittle:32:V8:T2`.
  The `Raw Binary` loader **silently ignores** `-loader-baseAddress` (it logs
  `WARN Skipping unsupported`), so the image loads at base `0x0`. `-process
  /arm9.bin` fails with "invalid filename specified" because of the leading `/`;
  use `-process` with no argument.
- Even with the entry seeded, Ghidra auto-analysis covered only 1723 instructions
  and 124 functions of a 465 KB ARM9. Below tens of thousands of instructions, a
  "zero references found" result means nothing.

Conclusion recorded so it is not re-derived: **route A needs a real Ghidra GUI
session with the `Actor` struct layout established from scratch** (vptr and
`Object` base members included). The decompilation is too early-stage to serve as
a layout anchor, and its own header comments are partly provisional.

## Emulator/bridge constraints

`bridge/desmume_bridge.lua` documents the working protocol. Hard constraints
learned the expensive way:

- `lua51.dll` was missing entirely; DeSmuME 0.9.13 loads it dynamically. Sourced
  from the local OBS install (`C:/Program Files/obs-studio/bin/64bit/lua51.dll`),
  verified 64-bit by reading the PE machine field (`0x8664`), copied next to
  `desmume.exe`. Nothing was downloaded for this.
- The Lua `input` table has only `get, popup, read, registerhotkey` — there is
  **no** input-writing API in this build.
- Clicking `Restart` or `Stop` in the Lua window **kills the whole emulator
  process** (reproduced twice). Load the script once per emulator run and control
  everything through the file protocol.
- The Lua script's own loop counter is not the game clock. Reading a flat RAM
  series and concluding "frozen address" was wrong twice; the second time the
  game was simply sitting in its pause menu. `status` now queries
  `emu.framecount()` / `emu.emulating()`, and every capture aborts unless
  `emulating == 1`.
- `player/*_LZ.bin` are **not** LZ-DS compressed despite the suffix: bytes 5-8
  are `BCA0`, the Nintendo skeletal-animation magic. The player's gravity is not
  in a data file; it is assigned in code, the way the decompilation assigns
  `accelV = -0x280` to a coin.

## Route B detector failures (2026-09-23), and a null that meant nothing

Four separate mistakes stacked up before the first credible number:

1. **Ranking candidates by longest monotone run selects eases, not jumps.** The
   winner, `0x02098210` (66 px amplitude), turned out to be a **linear** 297-frame
   excursion. A real jump is ~64 frames. Re-running the saved series through a
   parabola filter gave **0 arcs in all 15 survivors**, so the whole set was
   camera/platform easing. The physical signature to screen for is a *constant,
   non-zero second difference* (`arcs.curvature_prescreen`).
2. **The three scan windows overlapped.** `0x02080000 + 0x40000 == 0x020C0000`, so
   consecutive windows shared 128 KiB, and each address got scored against a
   different one-minute slice of the capture. Frame ranges were disjoint
   (31000-34557, 35342-38911, 39798-43359): the sweep was sequential, not parallel.
3. **A quadratic fit over a fixed frame window bleeds into the ground plateau**,
   which flattens the curvature: in planted data it turned -0.1875 into -0.1645 and
   blew the residual past the gate. The window has to be bounded by the motion.
4. **A parabola alone proves nothing** -- an ease toward a target fits just as
   well. The rule now is two independent estimators: position curvature and the
   velocity column's slope over the *same* descent windows must agree.

Then a full sweep of all 4 MiB (16 x 256 KiB) produced zero arc candidates with
`activity: ~2000 words changing` -- which looked like "wrong windows". It was
neither: the game had been sitting on the **world map** the whole time (Mario
unmoved across three screenshots, map timer counting 400 -> 338 -> 327), and the
framecount twice fell back to 26918, i.e. a savestate was reloaded mid-capture.
The likely reason no input reached the emulated DS: the floating
`desmume_bridge.lua` window had stolen focus, so keypresses went to the script
console instead of the game. **A capture is only as good as its precondition**;
`get_window_state` on the emulator costs one call and shows whether a stage is
even on screen.

## What is actually established

- ROM parsed; manifest of 2449 files in `tools/fs_manifest.json`.
- Level container structure across all 191 levels (above).
- Engine physics *model* from the CC0 decompilation: `velocity.y += accelV`,
  clamped by `minVelocity.y`; `position += velocity` in `Vec3_32`; scale 16.16
  (`0x8000 == 0.5`); worked example for a coin.
- Working collection bridge with a 60 Hz sampler and a 4 MiB RAM dumper.
- Route B (measuring gravity from screen-space frames, no addresses required) is
  the live next step, implemented in `bridge/capture_jump.py`.
