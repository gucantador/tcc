"""
Classifica imagens novas com um modelo já treinado por treinar_cnn.py.

Uso:
    python classificar.py foto.jpg
    python classificar.py foto1.jpg foto2.png pasta_com_imagens/
    python classificar.py foto.jpg --modelo resultados/treino_todas
    python classificar.py foto.jpg --salvar_ela        # salva o mapa ELA para visualizar

O tamanho do recorte e a qualidade do ELA são lidos do args.json salvo no
treino, para garantir que a imagem nova passe exatamente pelo mesmo processo.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comum import ela_de_arquivo
from treinar_cnn import CNNArtigo

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def listar_imagens(entradas):
    arquivos = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            arquivos += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        elif p.exists():
            arquivos.append(p)
        else:
            print(f"[AVISO] não encontrado: {e}")
    return arquivos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagens", nargs="+", help="arquivos de imagem e/ou pastas")
    ap.add_argument("--modelo", default="facebook/resultados/treino_todas",
                    help="pasta do experimento (com melhor_modelo.pt e args.json)")
    ap.add_argument("--limiar", type=float, default=0.5,
                    help="probabilidade de IA a partir da qual a imagem é classificada como IA")
    ap.add_argument("--salvar_ela", action="store_true",
                    help="salva o mapa ELA (amplificado) ao lado de cada imagem")
    args = ap.parse_args()

    pasta = Path(args.modelo)
    cfg = json.load(open(pasta / "args.json", encoding="utf-8"))
    tamanho, qualidade = cfg["tamanho"], cfg["qualidade"]

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = CNNArtigo(tamanho).to(disp)
    modelo.load_state_dict(torch.load(pasta / "melhor_modelo.pt", map_location=disp))
    modelo.eval()

    print(f"modelo: {pasta}  (recorte {tamanho}px, ELA qualidade {qualidade})\n")
    for arq in listar_imagens(args.imagens):
        try:
            formato = Image.open(arq).format
            ela = ela_de_arquivo(arq, tamanho, qualidade)
        except Exception as e:
            print(f"{arq.name}: erro ao abrir ({e})")
            continue
        if ela is None:
            print(f"{arq.name}: imagem menor que {tamanho}px, não dá para classificar")
            continue

        x = torch.from_numpy(ela.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            prob_ia = torch.softmax(modelo(x.to(disp)), dim=1)[0, 1].item()

        classe = "IA" if prob_ia >= args.limiar else "REAL"
        aviso = "  [atenção: arquivo não é JPEG]" if formato != "JPEG" else ""
        print(f"{arq.name}: {classe}  (probabilidade de IA: {prob_ia:.1%}){aviso}")

        if args.salvar_ela:
            vis = np.clip(ela.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            destino = arq.with_name(f"{arq.stem}_ela.png")
            Image.fromarray(vis).save(destino)


if __name__ == "__main__":
    main()
