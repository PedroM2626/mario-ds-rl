import sys, traceback
import ndspy.rom

path = sys.argv[1]
try:
    rom = ndspy.rom.NintendoDSRom(data=open(path, "rb").read())
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("== atributos do objeto ==")
attrs = [a for a in dir(rom) if not a.startswith("_")]
print(", ".join(attrs))

print("\n== escalares ==")
for a in attrs:
    try:
        v = getattr(rom, a)
    except Exception as e:
        continue
    if callable(v):
        continue
    if isinstance(v, (int, str, bytes)) or v is None:
        if isinstance(v, int) and v > 0xFF:
            print(f"  {a:<24} = 0x{v:X} ({v:,})")
        elif isinstance(v, bytes):
            print(f"  {a:<24} = {v[:16]!r} ({len(v)} bytes)")
        else:
            print(f"  {a:<24} = {v!r}")

print("\n== contêineres ==")
for a in attrs:
    try:
        v = getattr(rom, a)
    except Exception:
        continue
    if callable(v):
        continue
    if hasattr(v, "items"):
        print(f"  {a}: dict com {len(v)} chaves; amostra: {list(v.items())[:5]}")
    elif hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        print(f"  {a}: {type(v).__name__} com {len(v)} itens")

try:
    names = list(rom.filenames.keys())
    print(f"\n== filesystem: {len(names)} caminhos ==")
    for n in names[:60]:
        print("  ", n)
except Exception:
    traceback.print_exc()
