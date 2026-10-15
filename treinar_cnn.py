"""
Treina a CNN sobre os mapas ELA gerados por preparar_ela.py.

Arquitetura igual à de Martin-Rodriguez et al. (2023), Figura 4c,
adaptada do Matlab para PyTorch.

Exemplos:
    python treinar_cnn.py --treino original   # treina só com imagens originais
    python treinar_cnn.py --treino social     # treina só com imagens pós-rede social
    python treinar_cnn.py --treino ambos      # treina com as duas condições

O conjunto de TESTE sempre contém as duas condições, avaliadas separadamente.
Assim dá para medir quanto o desempenho cai depois da recompressão.

Dependências: pip install torch numpy
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- dados
def ler_index(caminho_csv):
    with open(caminho_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dividir_por_grupo(linhas, fr_val=0.15, fr_teste=0.15, seed=42):
    """Divide por GRUPO (a mesma imagem original + sua versão de rede social).
    Se a versão original cair no treino e a versão do WhatsApp no teste,
    a rede 'reconhece' a imagem em vez de aprender o padrão: vazamento."""
    grupos_por_rotulo = defaultdict(set)
    for l in linhas:
        grupos_por_rotulo[l["rotulo"]].add(l["grupo"])

    rng = random.Random(seed)
    destino = {}
    for rotulo, grupos in grupos_por_rotulo.items():
        grupos = sorted(grupos)
        rng.shuffle(grupos)
        n = len(grupos)
        n_teste, n_val = round(n * fr_teste), round(n * fr_val)
        for i, g in enumerate(grupos):
            split = "teste" if i < n_teste else "val" if i < n_teste + n_val else "treino"
            destino[(rotulo, g)] = split

    splits = defaultdict(list)
    for l in linhas:
        splits[destino[(l["rotulo"], l["grupo"])]].append(l)
    return splits


class ELADataset(Dataset):
    def __init__(self, linhas, aumentar=False):
        self.linhas = linhas
        self.aumentar = aumentar

    def __len__(self):
        return len(self.linhas)

    def __getitem__(self, i):
        l = self.linhas[i]
        x = np.load(l["caminho"]).astype(np.float32)        # H x W x 3
        x = torch.from_numpy(x).permute(2, 0, 1) / 255.0    # 3 x H x W
        if self.aumentar and random.random() < 0.5:
            x = torch.flip(x, dims=[2])                       # espelhamento não altera o ELA
        return x, int(l["rotulo"])


# ---------------------------------------------------------------- modelo
class CNNArtigo(nn.Module):
    """Figura 4c do artigo. Entrada 3x512x512 (ELA colorido)."""

    def __init__(self, canais=3):
        super().__init__()

        def bloco(c_in, c_out, k):
            return [nn.Conv2d(c_in, c_out, k, padding="same"),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            nn.BatchNorm2d(canais),            # normaliza a escala do ELA bruto
            *bloco(canais, 4, 5),              # Stage 1
            nn.MaxPool2d(5, stride=4),         # 512 -> 127
            *bloco(4, 8, 5),                   # Stage 2
            nn.MaxPool2d(4, stride=4),         # 127 -> 31
            *bloco(8, 16, 5),                  # Stage 3
            nn.MaxPool2d(2, stride=4),         # 31 -> 8
            *bloco(16, 32, 2),                 # Stage 4
        )
        self.classificador = nn.Linear(32 * 8 * 8, 2)

    def forward(self, x):
        return self.classificador(torch.flatten(self.features(x), 1))


# ---------------------------------------------------------------- métricas
def metricas(y_true, y_pred):
    """Positivo = IA (rótulo 1), como no artigo."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"n": total, "acuracia": (tp + tn) / total if total else 0.0,
            "precisao": p, "recall": r, "f1": f1,
            "matriz": [[tp, fn], [fp, tn]]}


@torch.no_grad()
def prever(modelo, loader, disp):
    modelo.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += modelo(x.to(disp)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return ys, ps


# ---------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cache", default="ela_cache")
    a.add_argument("--treino", choices=["original", "social", "ambos"], default="ambos")
    a.add_argument("--epocas", type=int, default=100)
    a.add_argument("--lote", type=int, default=16)
    a.add_argument("--lr", type=float, default=0.01)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--saida", default="resultados")
    args = a.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    disp = "cuda" if torch.cuda.is_available() else "cpu"

    linhas = ler_index(Path(args.cache) / "index.csv")
    splits = dividir_por_grupo(linhas, seed=args.seed)

    def filtra(ls):
        return ls if args.treino == "ambos" else [l for l in ls if l["condicao"] == args.treino]

    treino, val = filtra(splits["treino"]), filtra(splits["val"])
    teste = splits["teste"]                       # sempre as duas condições
    print(f"treino={len(treino)}  val={len(val)}  teste={len(teste)}  dispositivo={disp}")

    dl_treino = DataLoader(ELADataset(treino, aumentar=True), batch_size=args.lote, shuffle=True)
    dl_val = DataLoader(ELADataset(val), batch_size=args.lote)

    modelo = CNNArtigo().to(disp)
    otim = torch.optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9)  # SGDM, como no artigo
    perda_fn = nn.CrossEntropyLoss()

    saida = Path(args.saida) / f"treino_{args.treino}"
    saida.mkdir(parents=True, exist_ok=True)
    melhor_acc, historico = -1.0, []

    for ep in range(1, args.epocas + 1):
        modelo.train()
        soma, n = 0.0, 0
        for x, y in dl_treino:
            x, y = x.to(disp), y.to(disp)
            otim.zero_grad()
            perda = perda_fn(modelo(x), y)
            perda.backward()
            otim.step()
            soma += perda.item() * len(y); n += len(y)

        ys, ps = prever(modelo, dl_val, disp)
        acc_val = metricas(ys, ps)["acuracia"]
        historico.append({"epoca": ep, "perda_treino": soma / n, "acc_val": acc_val})
        print(f"época {ep:3d}  perda {soma / n:.4f}  acc_val {acc_val:.4f}")

        if acc_val > melhor_acc:
            melhor_acc = acc_val
            torch.save(modelo.state_dict(), saida / "melhor_modelo.pt")

    # ---- avaliação final no teste, separada por condição
    modelo.load_state_dict(torch.load(saida / "melhor_modelo.pt", map_location=disp))
    resultados = {"treino": args.treino, "melhor_acc_val": melhor_acc}
    for cond in ["original", "social"]:
        sub = [l for l in teste if l["condicao"] == cond]
        if not sub:
            continue
        ys, ps = prever(modelo, DataLoader(ELADataset(sub), batch_size=args.lote), disp)
        resultados[f"teste_{cond}"] = m = metricas(ys, ps)
        print(f"\nTESTE [{cond}]  acc {m['acuracia']:.3f}  P {m['precisao']:.3f}  "
              f"R {m['recall']:.3f}  F1 {m['f1']:.3f}  matriz [[TP FN][FP TN]] = {m['matriz']}")

    json.dump(resultados, open(saida / "metricas.json", "w"), indent=2, ensure_ascii=False)
    json.dump(historico, open(saida / "historico.json", "w"), indent=2)
    print(f"\nResultados salvos em {saida}/")


if __name__ == "__main__":
    main()
