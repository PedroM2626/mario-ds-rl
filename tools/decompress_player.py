"""Descompacta LZ-DS (magic 'LZ77') e procura tabelas de movimento do jogador.

LZ-DS e documentado e simples: cabecalho de 8 bytes ('LZ77', tam descompactado,
tam compactado) e depois blocos de 8 flags; flag bit=0 copia 1 byte literal,
bit=1 le um par de 2 bytes com back-reference em janela de 4 KiB. Implementado
aqui em vez de depender de lib, para o resultado ser auditavel.
"""
import os, struct, sys
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())


def lzds_decompress(data):
    if data[:4] != b"LZ77":
        return None
    out_len, comp_len = struct.unpack_from("<II", data, 4)
    src = 8
    dst = bytearray()
    while src < len(data) and len(dst) < out_len:
        flags = data[src]; src += 1
        for bit in range(8):
            if flags & (1 << bit):
                if src + 1 >= len(data):
                    return bytes(dst)
                d = data[src] | (data[src + 1] << 8); src += 2
                disp = (d & 0x0FFF) + 3
                ln = (d >> 12) + 3
                if disp > len(dst):
                    return bytes(dst)
                start = len(dst) - disp
                for k in range(ln):
                    dst.append(dst[start + k])
            else:
                if src >= len(data):
                    return bytes(dst)
                dst.append(data[src]); src += 1
                if len(dst) >= out_len:
                    break
    return bytes(dst)


for name in ["player/pl_LZ.bin", "player/pl2_LZ.bin", "player/plnovs_LZ.bin",
             "player/cap_LZ.bin"]:
    try:
        raw = rom.getFileByName(name)
    except Exception as e:
        print(f"{name}: nao encontrado ({e})"); continue
    print(f"\n===== {name}: {len(raw):,} bytes, magic={raw[:4]!r} =====")
    dec = lzds_decompress(raw)
    if dec is None:
        print("  nao e LZ-DS; primeiros bytes:", raw[:16].hex()); continue
    print(f"  descompactado: {len(dec):,} bytes")
    # abre NARC interno? muitos arquivos _LZ do NSMB envolvem um NARC
    print(f"  magic interno: {dec[:4]!r}")
    if dec[:4] == b"NARC":
        import ndspy.narc
        n = ndspy.narc.NARC(dec)
        print(f"  NARC com {len(n.files)} arquivos internos")
        for i, f in enumerate(n.files):
            print(f"    file {i}: {len(f):,} bytes  magic={f[:4]!r}")
        continue
    # procura sequencias de inteiros 16.16 monotonicos (tabelas de aceleracao)
    n = len(dec) // 4
    vals = struct.unpack(f"<{n}i", dec[:n*4])
    runs = []
    cur = 1
    for i in range(1, n):
        if vals[i] > vals[i-1] > 0 and vals[i] < 0x200000:
            cur += 1
        else:
            if cur >= 6:
                runs.append((i - cur, cur))
            cur = 1
    if cur >= 6:
        runs.append((n - cur, cur))
    print(f"  {len(runs)} runs de inteiros 16.16 crescentes (len>=6)")
    for off, ln in sorted(runs, key=lambda r: -r[1])[:6]:
        seg = vals[off:off+ln][:12]
        print(f"    +0x{off*4:05X} len={ln:<3} " + " ".join(f"{v/65536.0:.4f}" for v in seg))
