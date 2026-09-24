"""Testa hipoteses de layout com restricao universal sobre o corpus inteiro.

Para cada indice de secao, procura combos (delta, stride) tais que
(len - delta) % stride == 0 para TODOS os arquivos. Essa condicao e forte o
suficiente para eliminar coincidencias: um chute aleatorio raramente vale para
191 arquivos ao mesmo tempo.
"""
import struct, os, json, collections
import ndspy.rom

ROM = r"C:/Users/pedro/Downloads/desmume/Roms/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(data=open(ROM, "rb").read())
man = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_manifest.json"), encoding="utf-8"))
levels = sorted(r["path"] for r in man if r["path"].startswith("course/")
                and r["path"].endswith(".bin") and "_bgdat" not in r["path"])

def parse(name):
    blob = rom.getFileByName(name)
    hdr = struct.unpack_from("<I", blob, 0)[0]
    base = struct.unpack_from("<I", blob, 4)[0]
    pairs = [struct.unpack_from("<II", blob, off) for off in range(8, hdr - 7, 8)]
    # bloco lider, de `base` ate a primeira secao
    all_secs = [(base, pairs[0][0] - base)] + pairs
    return blob, all_secs

corpus = []
for name in levels:
    blob, secs = parse(name)
    corpus.append((name, blob, secs))
print(f"corpus: {len(corpus)} niveis, secoes por arquivo = {len(corpus[0][2])}\n")

NSEC = len(corpus[0][2])
print(f"{'sec':>3} {'ocorr':>5} {'tam: min..max':>16} {'unicos':>6}  hypotheses (delta+stride) validas p/ todos")
for i in range(NSEC):
    lens = [secs[i][1] for _, _, secs in corpus if i < len(secs) and secs[i][1] > 0]
    if not lens:
        print(f"{i:>3} {0:>5} {'-':>16} {0:>6}  sempre vazia"); continue
    hyps = []
    for stride in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        for delta in range(0, stride, 4):
            if all((l - delta) % stride == 0 for l in lens):
                hyps.append(f"{delta}+{stride}")
    print(f"{i:>3} {len(lens):>5} {f'{min(lens)}..{max(lens)}':>16} {len(set(lens)):>6}  "
          f"{', '.join(hyps) if hyps else 'NENHUMA (nao e vetor de registros fixos)'}")

print("\n== correlacao com a extensao do nivel (proxy de qtd de objetos) ==")
ref = max(range(NSEC), key=lambda i: len(set(s[i][1] for _, _, s in corpus if i < len(s))))
for i in range(NSEC):
    lens = [s[i][1] for _, _, s in corpus if i < len(s)]
    print(f"  sec {i}: amplitude {max(lens)-min(lens):>6}  mediana {sorted(lens)[len(lens)//2]:>6}")
