"""
Varre as pastas definidas em config_dataset.py e gera index.csv.

Não calcula ELA (isso é feito durante o treino). Só lê o cabeçalho de cada
imagem para registrar formato e tamanho, e imprime relatórios para encontrar
vieses ANTES de treinar:
  - formato real dos arquivos por classe/condição (PNG vs JPEG)
  - quantas imagens são menores que o recorte (e seriam descartadas)
  - quantas imagens originais têm o par compartilhado em cada rede

Uso:
    python indexar.py --tamanho 256 --saida index.csv
"""

import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

import config_dataset as cfg

ROTULOS = {"real": 0, "ia": 1}
ORIENTACOES_ROTACIONADAS = {5, 6, 7, 8}    # EXIF que trocam largura/altura


def expandir_fontes():
    """Transforma as entradas do config (com curingas) em pastas concretas."""
    entradas = []
    for f in cfg.FONTES:
        pastas = sorted(p for p in glob.glob(f["pasta"]) if os.path.isdir(p))
        if not pastas:
            print(f"[AVISO] nenhuma pasta encontrada para: {f['pasta']}")
        for p in pastas:
            entradas.append({
                "pasta": p,
                "rotulo": ROTULOS[f["rotulo"]],
                "condicao": f["condicao"],
                "fonte": f["fonte"] or os.path.basename(os.path.normpath(p)),
            })
    return entradas


def id_grupo(nome_sem_ext):
    if cfg.GRUPO_REGEX:
        m = re.search(cfg.GRUPO_REGEX, nome_sem_ext)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return nome_sem_ext


def ler_cabecalho(args):
    caminho, entrada = args
    try:
        with Image.open(caminho) as im:
            w, h = im.size
            if im.getexif().get(274) in ORIENTACOES_ROTACIONADAS:
                w, h = h, w
            formato = im.format
    except Exception as e:
        return None, f"{caminho}: {e}"
    # grupo = subpasta relativa + id do arquivo (ex: "animals/0001"), para que
    # arquivos com o mesmo nome em categorias diferentes não se misturem
    rel = Path(os.path.relpath(caminho, entrada["pasta"]))
    sub = rel.parent.as_posix()
    gid = id_grupo(rel.stem)
    grupo = gid if sub == "." else f"{sub}/{gid}"
    return {
        "caminho": Path(caminho).as_posix(),
        "rotulo": entrada["rotulo"],
        "condicao": entrada["condicao"],
        "fonte": entrada["fonte"],
        "grupo": grupo,
        "formato": formato,
        "largura": w,
        "altura": h,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="index.csv")
    ap.add_argument("--tamanho", type=int, default=256,
                    help="tamanho do recorte que será usado no treino (só para o relatório)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    entradas = expandir_fontes()
    if not entradas:
        print("Nenhuma fonte configurada. Edite FONTES em config_dataset.py.")
        return

    tarefas = []
    for e in entradas:
        for raiz, _, arquivos in os.walk(e["pasta"]):
            for a in arquivos:
                if os.path.splitext(a)[1].lower() in cfg.EXTENSOES:
                    tarefas.append((os.path.join(raiz, a), e))
    print(f"{len(tarefas)} arquivos encontrados em {len(entradas)} pastas. Lendo cabeçalhos...")

    linhas, erros = [], []
    with Pool(args.workers) as pool:
        for i, (linha, erro) in enumerate(pool.imap_unordered(ler_cabecalho, tarefas, chunksize=256)):
            (linhas.append(linha) if linha else erros.append(erro))
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{len(tarefas)}")

    campos = ["caminho", "rotulo", "condicao", "fonte", "grupo", "formato", "largura", "altura"]
    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(sorted(linhas, key=lambda l: l["caminho"]))
    print(f"\n{len(linhas)} imagens indexadas em {args.saida} ({len(erros)} erros)")
    for e in erros[:10]:
        print("  [ERRO]", e)

    nome = {0: "real", 1: "ia"}

    # ---------- relatório 1: formato
    print("\n=== FORMATO REAL DOS ARQUIVOS (procure diferenças entre real e ia) ===")
    c = Counter((nome[l["rotulo"]], l["condicao"], l["formato"]) for l in linhas)
    for (r, cond, fmt), n in sorted(c.items()):
        print(f"  {r:5s} {cond:10s} {str(fmt):6s} {n}")

    # ---------- relatório 2: tamanho
    print(f"\n=== IMAGENS MENORES QUE {args.tamanho}px (seriam descartadas no treino) ===")
    total = Counter((nome[l["rotulo"]], l["condicao"]) for l in linhas)
    peq = Counter((nome[l["rotulo"]], l["condicao"]) for l in linhas
                  if min(l["largura"], l["altura"]) < args.tamanho)
    for k in sorted(total):
        pct = 100 * peq[k] / total[k]
        alerta = "  <-- ATENÇÃO" if pct > 10 else ""
        print(f"  {k[0]:5s} {k[1]:10s} {peq[k]:7d} de {total[k]:7d} ({pct:5.1f}%){alerta}")

    # ---------- relatório 3: pareamento original <-> compartilhada
    if len({l["condicao"] for l in linhas}) < 2:
        print("\n(só uma condição configurada: relatório de pareamento omitido)")
        return
    print("\n=== PAREAMENTO: originais que têm versão em cada condição ===")
    por_grupo = defaultdict(set)
    for l in linhas:
        por_grupo[(l["rotulo"], l["fonte"], l["grupo"])].add(l["condicao"])
    com_original = [conds for conds in por_grupo.values() if "original" in conds]
    condicoes = sorted({l["condicao"] for l in linhas} - {"original"})
    for cond in condicoes:
        n = sum(cond in conds for conds in com_original)
        print(f"  {cond:10s} {n} de {len(com_original)} originais têm par")
    orfas = sum(1 for conds in por_grupo.values() if "original" not in conds)
    if orfas:
        print(f"  [AVISO] {orfas} grupos sem versão original: confira GRUPO_REGEX "
              f"e se a 'fonte' é a mesma em todas as condições.")


if __name__ == "__main__":
    main()
