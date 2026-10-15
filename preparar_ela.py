"""
Gera os mapas ELA de todo o dataset e salva em .npy para o treino da CNN.

Estrutura esperada (o mesmo nome de arquivo nas pastas original/ e social/):

    dataset/
        real/original/foto001.jpg
        real/social/foto001.jpg      <- mesma foto após upload/download
        ia/original/img001.png
        ia/social/img001.jpg

Saída:
    ela_cache/<classe>/<condicao>/<nome>.npy   (uint8, 512x512x3, diferença bruta)
    ela_cache/index.csv                        (caminho, classe, condicao, grupo, formato)
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLASSES = {"real": 0, "ia": 1}
CONDICOES = ["original", "social"]


def carregar_rgb(caminho):
    img = Image.open(caminho)
    formato = img.format                      # formato REAL, não a extensão
    img = ImageOps.exif_transpose(img)         # corrige rotação pelo EXIF
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fundo = Image.new("RGB", img.size, (255, 255, 255))
        fundo.paste(img, mask=img.split()[-1])
        img = fundo
    return img.convert("RGB"), formato


def recorte_central(img, tamanho):
    w, h = img.size
    if min(w, h) < tamanho:
        return None
    esq, topo = (w - tamanho) // 2, (h - tamanho) // 2
    return img.crop((esq, topo, esq + tamanho, topo + tamanho))


def calcular_ela(img, qualidade=95):
    """Diferença absoluta entre a imagem e sua recompressão JPEG.
    Sem normalização por imagem: o NÍVEL do erro é justamente a informação
    que a rede precisa ver."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=qualidade)
    buf.seek(0)
    recomprimida = Image.open(buf).convert("RGB")
    a = np.asarray(img, dtype=np.int16)
    b = np.asarray(recomprimida, dtype=np.int16)
    return np.abs(a - b).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--saida", default="ela_cache")
    p.add_argument("--tamanho", type=int, default=512)
    p.add_argument("--qualidade", type=int, default=95)
    args = p.parse_args()

    raiz, saida = Path(args.dataset), Path(args.saida)
    linhas, pulados = [], 0
    contagem_formato = {}

    for classe in CLASSES:
        for cond in CONDICOES:
            pasta = raiz / classe / cond
            if not pasta.exists():
                continue
            destino = saida / classe / cond
            destino.mkdir(parents=True, exist_ok=True)
            for arq in sorted(pasta.iterdir()):
                if arq.suffix.lower() not in EXTS:
                    continue
                try:
                    img, formato = carregar_rgb(arq)
                except Exception as e:
                    print(f"[ERRO] {arq}: {e}")
                    continue

                # ELA na imagem inteira e só depois o recorte: assim a grade
                # 8x8 do JPEG original continua alinhada com a da recompressão
                ela_full = calcular_ela(img, args.qualidade)
                crop = recorte_central(Image.fromarray(ela_full), args.tamanho)
                if crop is None:
                    pulados += 1
                    continue
                ela = np.asarray(crop)

                out = destino / f"{arq.stem}.npy"
                np.save(out, ela)
                linhas.append([out.as_posix(), CLASSES[classe], cond, arq.stem, formato])
                chave = (classe, cond, formato)
                contagem_formato[chave] = contagem_formato.get(chave, 0) + 1

    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caminho", "rotulo", "condicao", "grupo", "formato"])
        w.writerows(linhas)

    print(f"\n{len(linhas)} mapas ELA salvos | {pulados} imagens menores que {args.tamanho}px puladas")
    print("\nFormato real dos arquivos por grupo (confira se real e IA estão equilibrados!):")
    for (classe, cond, fmt), n in sorted(contagem_formato.items()):
        print(f"  {classe:5s} {cond:9s} {fmt:5s} {n}")


if __name__ == "__main__":
    main()
