"""Desmontador de traversal recursivo (worklist) sobre capstone.

Por que: desmontagem LINEAR produz lixo. A varredura anterior devolveu dezenas
de LDC/STC/VST --- instrucoes de coprocessador que o ARM9 do DS nao tem --
porque o capstone andou por cima de pools de literais e fronteiras de funcao.

Aqui: comeca no entry point, segue bl/b/blx, usa o bit 0 do alvo para escolher
ARM vs Thumb, e onde a decodificacao falha ou e obviamente invalida avanca como
dado (pool de literais). So o que e alcancavel por chamadas virou codigo.
"""
import struct, sys, collections
import ndspy.rom
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, CS_MODE_LITTLE_ENDIAN

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
arm9 = bytes(rom.arm9)
LOAD, ENTRY = rom.arm9RamAddress, rom.arm9EntryAddress
END = LOAD + len(arm9)

md_arm = Cs(CS_ARCH_ARM, CS_MODE_ARM | CS_MODE_LITTLE_ENDIAN)
md_arm.detail = False
md_th = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_LITTLE_ENDIAN)
md_th.detail = False

BAD_ARM = ("ldc", "stc", "udf", "bkpt", "svc")
BAD_TH = ("udf", "bkpt", "svc", "pop.w", "ldr.w")   # ldr.w mantido: e valido


def inside(a):
    return LOAD <= a < END


def decode_one(addr, thumb):
    """Retorna (instrucao, tamanho) ou (None, 0) se nao decodifica bem."""
    md = md_th if thumb else md_arm
    n = 4
    buf = arm9[addr - LOAD: addr - LOAD + (4 if not thumb else 4)]
    if len(buf) < 2:
        return None, 0
    try:
        it = iter(md.disasm(buf, addr))
        ins = next(it)
    except Exception:
        return None, 0
    if ins.mnemonic.startswith(BAD_ARM if not thumb else ("ldc", "stc", "udf")):
        return None, 0
    return ins, ins.size


seen = {}                     # addr -> (mnemonic, op_str, thumb)
work = collections.deque([(ENTRY & ~1, bool(ENTRY & 1))])
# Thumb e ARM se misturam; o bit 0 do endereco marca o modo
work.append((0x02000801, True))

steps = 0
while work and steps < 400000:
    steps += 1
    addr, thumb = work.popleft()
    addr &= ~1
    if not inside(addr) or addr in seen:
        continue
    ins, size = decode_one(addr, thumb)
    if ins is None:
        # provavel pool de literais: avanca e tenta retomar
        nxt = addr + (2 if thumb else 4)
        if inside(nxt):
            work.append((nxt, thumb))
        continue
    seen[addr] = (ins.mnemonic, ins.op_str, thumb)
    m, ops = ins.mnemonic, ins.op_str

    # follow branches
    def targets(txt):
        """Extrai alvos de ramo. O capstone escreve imediatos com '#'
        (ex.: 'b #0x2000820'), entao o '#' precisa ser removido antes do
        int(...,16) -- sem isso nenhum ramo era seguido e o worklist morria."""
        out = []
        for tok in txt.replace(",", " ").split():
            t = tok.lstrip("#")
            if t.startswith("0x"):
                try:
                    v = int(t, 16)
                except ValueError:
                    continue
                if inside(v):
                    out.append(v)
        return out

    if m.startswith("bl") or m == "b" or m.startswith("b.") or m.startswith("cbz") or m.startswith("cbnz"):
        for t in targets(ops):
            work.append((t, bool(t & 1)))
            break
        if m.startswith("b.") or m.startswith("cb"):
            work.append((addr + size, thumb))
    elif m.startswith("b"):                      # ramos condicionais ARM
        work.append((addr + size, thumb))
        for t in targets(ops):
            work.append((t, bool(t & 1)))
    elif m in ("bx", "blx") and "lr" not in ops:
        work.append((addr + size, thumb))        # chamada indireta: nao seguimos
    else:
        work.append((addr + size, thumb))

print(f"instrucoes alcancadas por chamada: {len(seen):,} em {steps:,} passos de worklist")
th = sum(1 for v in seen.values() if v[2])
print(f"  modo Thumb: {th:,}   modo ARM: {len(seen)-th:,}")

OBJ = __import__("re").compile(r"\[(r\d+|sl|fp),\s*#(0x[0-9a-f]{2,3})\]")
by_off = collections.Counter()
where = collections.defaultdict(list)
for addr, (m, ops, thumb) in seen.items():
    if "sp" in ops.split("[")[-1][:4] or "[pc" in ops or "[sp" in ops:
        continue
    mm = OBJ.search(ops)
    if mm:
        off = int(mm.group(2), 16)
        by_off[off] += 1
        where[off].append((addr, f"{m} {ops}"))

print(f"\n== acessos a [rN,#offset] no codigo alcancavel (top 15 offsets) ==")
for off, c in by_off.most_common(15):
    print(f"  +0x{off:03X}: {c:>5}")
print("\n== alvo: offsets da area de fisica do Actor ==")
for off in (0xB4, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC):
    lst = where.get(off, [])
    print(f"  +0x{off:03X}: {len(lst)}")
    for addr, txt in sorted(lst)[:10]:
        print(f"       0x{addr:08X}: {txt}")
