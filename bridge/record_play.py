"""Grava o estado da RAM enquanto VOCE joga, para depois eu localizar enderecos.

Uso:
    python record_play.py              # 30 despejos, ~3s entre eles
    python record_play.py 60 1.5       # 60 despejos, 1.5s entre eles

Enquanto roda, basta jogar normalmente na janela do DeSmuME. Nao precisa fazer
nada sincronizado: o que eu preciso e de variacao (andar, correr, pular curto,
pular longo, cair, apanhar), nao de momentos especificos.

Os arquivos vao para ramdir/ e o log de frames para ramdir/log.csv.
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsmb_env

OUTDIR = os.path.join(HERE, "ramdir")
os.makedirs(OUTDIR, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
BASE, SIZE = 0x02000000, 0x400000

b = nsmb_env.Bridge()
print("[ping]", b.send("ping", timeout=15))
print(f"gravando {N} despejos de {SIZE/1024/1024:.0f} MiB com {GAP}s de intervalo")
print(">>> jogue agora na janela do DeSmuME. Ctrl+C para parar. <<<\n")

log = open(os.path.join(OUTDIR, "log.csv"), "w")
log.write("idx,tag,frame,wallclock\n")

got = 0
try:
    for i in range(N):
        t0 = time.time()
        tag = "r%03d" % i
        try:
            blob = b.dump(tag, BASE, SIZE)
        except nsmb_env.BridgeError as e:
            print(f"  [{i}] falhou: {e}")
            time.sleep(1)
            continue
        # o arquivo ram_<tag>.bin ja foi escrito pelo Lua; registramos o frame
        frame = -1
        try:
            frame = int(b.send("ping", timeout=15).split()[-1])
        except Exception:
            pass
        log.write(f"{i},{tag},{frame},{time.strftime('%H:%M:%S')}\n")
        log.flush()
        got += 1
        print(f"  [{i:>3}] {len(blob)/1048576:.1f} MiB  frame={frame}")
        dt = time.time() - t0
        if i < N - 1:
            time.sleep(max(0.1, GAP - dt))
finally:
    log.close()

print(f"\ngravado: {got} despejos em {OUTDIR}")
print("proximo passo: python analyze_play.py")
