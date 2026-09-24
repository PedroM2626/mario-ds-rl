# Academic Reverse Engineering Report: *New Super Mario Bros.* (Nintendo DS)

**Technical Report NTR-A2DP-EUR-RE-01**  
*Author:* Reinforcement Learning & Reverse Engineering Team  
*Date:* September 2026  
*Target Binary:* `0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds` (CRC32: `0x7896C7C1`, NTR-A2DP-EUR)  
*Decompilation Anchor Reference:* `NSMB-Decomp/nsmb` (CC0 Clean-Room Decompilation)  

---

## Abstract

This report documents the reverse engineering of the Nintendo DS title *New Super Mario Bros.* (EUR release NTR-A2DP-EUR) to support physics-informed world models, offline dataset generation, and direct RAM-telemetry reinforcement learning. We report both verified structural truths and negative results (falsified hypotheses). Crucially, we prove the exact binary framing of level entity records (12-byte records terminated by a `0xFFFFFFFF` sentinel across 189/189 non-empty stages), correct a 16× fixed-point scaling error in prior literature by establishing the engine's native 20.12 fixed-point arithmetic ($0x1000 = 1.0\text{ px}$), map 342 actor profile IDs to their C++ execution classes, detail emulator bridge telemetry constraints in DeSmuME, and identify the exact level-clear stage flag at `0x020DCF27`. Finally, we document remaining reverse-engineering gaps, specifically the missing player-specific jump state machine in static disassembly.

---

## 1. ROM Architecture & File System Extraction

### 1.1 ROM Encryption Status (Falsified KEY1 Hypothesis)
* **Initial Hypothesis:** The ROM was hypothesized to require KEY1/`headcrypto` decryption due to the apparent absence of standard Nitro containers such as `NCCL`, `NSCR`, or `YSARC`.
* **Empirical Falsification:** Direct binary inspection revealed runs of 5,696 contiguous null bytes (`0x00`) and plaintext ASCII string tables (`.bin`, `arc`, `nitro`) inside the ARM9 binary region. A block/stream cipher cannot produce contiguous zero runs of this length.
* **Ground Truth:** The European ROM is **unencrypted** (`encryptionSeedSelect = 0`). The ARM9 binary loads directly at entry point `0x02000800`.

### 1.2 File System Inventory
Using `ndspy.rom.NintendoDSRom`, the ROM unpacks into 2,449 named assets spanning 22.6 MiB of content across 120 NARC archives:

| Asset Directory | File Count | Content Description |
|---|---|---|
| `course/` | 382 | 191 stage binaries (`A##_#.bin`) paired with 191 terrain collision maps (`*_bgdat.bin`) |
| `enemy/` | 368 | Skeletal animations (`.nsbca`), 3D models (`.nsbmd`), textures (`.nsbtx`) |
| `BG_chk/` | 55 | Level chunk definition tables (`*MainUnitChangeData.bin`) |
| `BG_nsc/` | 118 | Screen background tilemaps |
| `player/` | 4 | Player asset packages (`pl_LZ.bin`, `pl2_LZ.bin`, `plnovs_LZ.bin`, `cap_LZ.bin`) |
| `map/` | 8 | World map models (`w1.nsbmd` – `w8.nsbmd`) |

*Note on Compression:* Files suffixed with `_LZ` in `player/` are **not** LZ77/LZ-DS compressed; bytes 5–8 contain `BCA0` (skeletal animation header).

---

## 2. Level Container Binary Layout (`course/*.bin`)

### 2.1 Container Header Layout
Every stage file `course/X##_#.bin` contains a 112-byte (`0x70`) header consisting of:
1. `u32` header size (`0x00000070`).
2. `u32` configuration word at `0x04` (`0x00000020` in all observed stages).
3. 13 contiguous `(offset: u32, length: u32)` descriptor pairs starting at byte `0x08`, exactly tiling the binary to the terminal byte.

### 2.2 Section GCD Stride Analysis
Computing the Greatest Common Divisor (GCD) of section lengths across all 191 levels proved invariant record strides:

| Section Index | Byte Offset in Header | Stride | Universal Invariance | Semantic Content |
|---|---|---|---|---|
| `sec0` | `0x08` | 24 B | 191/191 levels | Camera bounding regions / zones |
| `sec1`–`sec3` | `0x18`–`0x28` | 20 B | 191/191 levels | Screen transitions, path nodes |
| `sec4` | `0x30` | 20 B | 191/191 levels | Geometry vectors ($X, Y \in [0, 5840]$, multiples of 16) |
| `sec5` | `0x38` | **12 B** | 189/191 levels | **Entity / Placed Actor Stream** |
| `sec6` | `0x40` | 16 B | 191/191 levels | Ambient triggers / sound zones |
| `sec8` | `0x50` | 8 B | 134/191 levels | Secondary path links |
| `sec10` | `0x60` | 16 B | 191/191 levels | Tile event triggers |
| `sec12` | `0x70` | 16 B | 189/191 levels | Stage metadata footer (single struct) |

### 2.3 The Entity Stream (`sec5` / Block 6) Structural Proof
* **The Negative Result (The 4-Byte Shift Trap):** An earlier analysis hypothesized a 4-byte header at offset 0 followed by 12-byte records. This shifted every unpacked field by two 16-bit words, causing 37% of extracted entity coordinates to evaluate as zeros.
* **The Exact Proof:** The 12-byte records begin at **offset 0**. The extra 4 bytes at the end of the section represent a **terminal sentinel word `0xFFFFFFFF` (`-1`)**.
  - Exactly 189 of 191 levels satisfy `(length - 4) % 12 == 0`.
  - In all 189 levels with non-empty entity streams, the final `u32` word is identically `0xFFFFFFFF`.
  - The remaining 2 levels (`B01_3.bin` and `G05_2.bin`) have `length == 4`, representing empty entity lists consisting solely of the `0xFFFFFFFF` terminator.

### 2.4 Entity Record Format
Each 12-byte record is structured as six unsigned 16-bit words (`<6H`):

$$\text{Record} = \left( \text{id}, x_{\text{tile}}, y_{\text{tile}}, w_3, \text{settings}, w_5 \right)$$

* $\text{id}$ (`u16`): Actor profile identifier ($\le 384$). 84.5% resolve directly to known actor classes.
* $x_{\text{tile}}$ (`u16`): Horizontal position in 16-pixel tile units. Across the 5,311 entities, $x_{\text{tile}} \in [0, 447]$ (pixel span $0$ to $7,152\text{ px}$, median $1,216\text{ px}$).
* $y_{\text{tile}}$ (`u16`): Vertical position in 16-pixel tile units ($y_{\text{tile}} \in [0, 207]$, pixel span $0$ to $3,312\text{ px}$).
* $w_3$ (`u16`): Depth/layer or subtype parameter (99.7% of values $\le 64$; maximum 13,056).
* $\text{settings}$ (`u16`): Actor spawn configuration flags.
* $w_5$ (`u16`): Optional link index or secondary parameter ($65535 = \text{none}$).
* **Pixel Conversion:** $x_{\text{px}} = x_{\text{tile}} \times 16$, $y_{\text{px}} = y_{\text{tile}} \times 16$. With this framing, only 36 out of 5,311 entities (0.68%) reside at $(0, 0)$.

---

## 3. Engine Kinematics & Physics Formalism

### 3.1 Fixed-Point Representation: 20.12 vs. 16.16
* **Falsified Assumption:** Early pipeline iterations assumed standard Nintendo 16.16 fixed point ($0x10000 = 1.0\text{ px}$). This produced a $16\times$ discrepancy in velocity integration.
* **Mathematical Proof:** Source code analysis of `NSMB-Decomp/nsmb` confirmed the engine utilizes **20.12 fixed point ($0x1000 = 4096 = 1.0\text{ px}$)**:
  - `src/AAA.hpp:17`: `#define _FixedFlt(flt) ((i32)(flt * 4096.0))`
  - `src/AAA.hpp:19-23`: `_FixedMul` shifts right by 12 bits (`>> 12`).
  - `src/Bases/StageEntity.cpp:100-101`: `viewOffset.x << 0xc` converts pixels to world coordinates.
  - `src/Bases/Coin.cpp:517-518`: Half-tile offset is `0x8000` ($8\text{ px} \times 4096$), full tile is `0x10000` ($16\text{ px} \times 4096$).

### 3.2 Discrete Equations of Motion
In `src/Bases/Actor.cpp`, the vertical velocity update is evaluated per frame (60 Hz):

$$v_{y, t+1} = \max\left( v_{y, t} + a_v, \; v_{\min, y} \right)$$

$$p_{y, t+1} = p_{y, t} + v_{y, t+1}$$

* Sign convention: Upward vertical velocity is positive; gravity $a_v$ is negative.
* Lower bound: $v_{\min, y}$ operates as a **one-sided lower clamp** representing terminal fall velocity; it does not restrict upward jump velocity.

### 3.3 Extracted Physical Constants

| Class / Context | Parameter | Raw Fixed-Point | $\text{px/frame}$ or $\text{px/f}^2$ | Physical Unit (60 Hz) | Source Reference |
|---|---|---|---|---|---|
| `StageEntity` (Default) | $a_v$ (gravity) | `-0x300` | $-0.1875\text{ px/f}^2$ | $-675.0\text{ px/s}^2$ | `StageEntity.cpp:13` |
| `StageEntity` (Default) | $v_{\min, y}$ (terminal fall) | `-0x4000` | $-16.0\text{ px/f}$ | $-960.0\text{ px/s}$ | `StageEntity.cpp:15` |
| `StageEntity::_18` (Shell) | $a_h$ (friction/accel) | `0x100` | $0.0625\text{ px/f}^2$ | $225.0\text{ px/s}^2$ | `StageEntity.cpp:321` |
| `Coin` (Spawn State) | $a_v$ | `-0x280` | $-0.15625\text{ px/f}^2$ | $-562.5\text{ px/s}^2$ | `Coin.cpp:256` |
| `Coin` (Terminal Fall) | $v_{\min, y}$ | `-0x8000` | $-32.0\text{ px/f}$ | $-1920.0\text{ px/s}$ | `Coin.cpp:254` |
| `PlayerBase` (Mini Mario) | Gravity Multiplier | `0xD00` | $\times 0.8125$ | $\times 0.8125$ | `PlayerBase.cpp:1102` |

---

## 4. Actor Profile Mapping

In NSMB DS, actors are instantiated dynamically through `CurrentProfileTable[object_id]->constructor()`.
By scraping profile declarations `ActorProfile X_Profile = { Class::create, <id>, ... }` from `NSMB-Decomp`, **342 profile slots** were identified and exported to `out/nsmb_object_ids.json`:

* 320 IDs agree exactly with clean-room header annotations.
* 22 IDs were resolved via constructor pointer cross-referencing.
* **Missing Profiles:** Goomba (`0xA0`), Koopa (`0xA3`), and Piranha Plant (`0x1F`) do not have static profile entries in the decompilation tree because they are linked as native overlay modules or core engine entities.

---

## 5. Emulator Telemetry & RAM State Architecture

### 5.1 Validated Memory Addresses (EUR Release NTR-A2DP-EUR)

| Variable | Address | Type | Validation Methodology |
|---|---|---|---|
| Lives Counter | `0x0209DC00` | `u8` | Memory decrement on death; matches Action Replay `2209DC00` |
| Camera X Coordinate | `0x02098240`, `0x020DCFA0` | `u32` | Verified deadzone tracking and freeze on world map |
| Horizontal Odometer | `0x0209AE9C` | `u32` | Monotonically increments during forward rightward motion |
| Mario Base Address | `0x021C1890` | Pointer | Validated across savestates; points to actor with type `0x1C` at `+0x44` |
| Mario Vertical Position ($Y$) | `0x021C1890 + 0x60` | `s32` | 20.12 fixed point; baseline floor is $-480\text{ px}$ |
| Mario Horizontal Position ($X$) | `0x021C1890 + 0x68` | `s32` | 20.12 fixed point; tracks screen traversal |
| Entity Linked List Node | `0x021C1890 + 0x38` | Pointer | Circular doubly-linked list (`[prev: u32, next: u32]`) |
| Stage Clear Flag | `0x020DCF27` | `u8` | Transitions from 0 to 1 upon flagpole descent / castle entry; verified via differential RAM search |

### 5.2 Dynamic Entity Scanning
Enemies dynamically insert into the active linked list when within approximately 20 steps of the camera viewport. The environment reader traverses the memory span `0x021C0000`–`0x021E0000`, walks the active nodes, and extracts the 3 closest entities relative to Mario:

$$s_{\text{enemy}, i} = \left( \Delta x_i, \; \Delta y_i, \; \text{type}_i \right)$$

### 5.3 Player Jump State Machine Decomposition (Route B Empirical Telemetry)

Using high-frequency $60.0\text{ Hz}$ single-frame memory sampling on DeSmuME ([`tools/decompose_mario_jump.py`](../tools/decompose_mario_jump.py)), we captured and decomposed Mario's vertical jump physics across 4 distinct control regimes, resolving Route B:

| Jump Regime | Input Sequence | Airtime ($t_{\text{air}}$) | Peak Rise ($h_{\max}$) | Launch $v_{y, 0}$ | $g_{\text{ascent}}$ (hold) | $g_{\text{descent}}$ (fall) | Apex Hang | Horizontal Reach |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tap Jump (Short Hop)** | B held 2 frames | 22 frames | $19.94\text{ px}$ | $3.5938\text{ px/f}$ | $0.3284\text{ px/f}^2$ | $0.2662\text{ px/f}^2$ | 3 frames | $0.0\text{ px}$ |
| **Full Standing Jump** | B held 40 frames | 48 frames | $63.81\text{ px}$ | $3.5938\text{ px/f}$ | $0.0963\text{ px/f}^2$ | $0.2339\text{ px/f}^2$ | 3 frames | $0.0\text{ px}$ |
| **Walking Jump** | Right + B held | 52 frames | $71.06\text{ px}$ | $3.7188\text{ px/f}$ | $0.0919\text{ px/f}^2$ | $0.2118\text{ px/f}^2$ | 3 frames | $75.33\text{ px}$ |
| **Running Leap** | Right + Y (Dash) + B | 52 frames | $71.06\text{ px}$ | $3.7188\text{ px/f}$ | $0.0919\text{ px/f}^2$ | $0.2118\text{ px/f}^2$ | 3 frames | $109.73\text{ px}$ |

*Key Findings:*
1. **Variable Jump Height Mechanics:** Sustained button hold reduces upward gravity deceleration by $3.57\times$ ($0.0919\text{ px/f}^2$ vs $0.3284\text{ px/f}^2$), extending the jump ascent from 10 frames to 25 frames.
2. **Horizontal Boost Coupling:** Running with Dash (`KEY_Y`) increases vertical takeoff velocity from $3.5938$ to $3.7188\text{ px/frame}$ ($+3.48\%$), yielding an extra $7.25\text{ px}$ of vertical clearance and enabling $109.73\text{ px}$ of horizontal coverage.
3. **Terminal Downward Velocity:** Capped at $v_{y, \text{term}} = -4.0\text{ px/frame}$ ($16,384$ in 20.12 fixed point).

---

## 6. Scientific Negative Results & Falsified Methodologies

Documenting negative results is essential to prevent future research cycles from re-exploring unproductive paths:

1. **Linear Capstone Disassembly on ARM9:**
   - Attempting linear disassembly over raw ARM9 binary produced spurious `LDC`, `STC`, and `VSTR` instructions. The DS ARM946E-S has no floating-point coprocessor; the disassembler was interpreting inline literal pools and alignment padding as code.
2. **Recursive Function Worklist Termination:**
   - A basic recursive disassembler halted after ~70 instructions because Nitro SDK boot code switches processor state via Thumb interworking (`msr cpsr_c, r0`) and indirect register jumps (`bx r12`), which cannot be followed without full control-flow recovery.
3. **Headless Ghidra 12 Analysis Gaps:**
   - Ghidra's headless auto-analysis without manual type seeding analyzed only 1,723 instructions (124 functions) out of the 465 KB ARM9 binary. "Zero data references found" was an artifact of incomplete decompilation coverage, not evidence of absence.
4. **Monotone Run Heuristics for Coordinate Discovery:**
   - Searching 4 MiB of RAM for monotonic variables selected camera interpolation easings (`0x02098210`), not player coordinates. A true physical ballistic jump must exhibit a constant non-zero second difference ($\Delta^2 y \approx \text{const}$).
5. **DeSmuME Lua Bridge Focus Contention:**
   - In DeSmuME 0.9.13, opening the Lua execution console steals OS window focus. Keypresses sent via synthetic automation or human input are swallowed by the console rather than routed to the emulated DS core.

---

## 7. Outstanding Reverse Engineering Gaps

1. **Chunk Terrain Collision Decoding (`BG_chk`):**
   - While `_bgdat.bin` provides bounding boxes for flat ground, composite tiles (slopes, pipes, question blocks) reference indices into `BG_chk/*MainUnitChangeData.bin` which remain undecoded.


