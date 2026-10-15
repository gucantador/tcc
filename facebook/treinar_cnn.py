"""
Treina a CNN do artigo base (Martin-Rodriguez et al., 2023) sobre mapas ELA
calculados durante o treino, a partir do index.csv gerado por indexar.py.

Exemplos:
    # treina só com originais, testa em todas as condições
    python treinar_cnn.py --treino_condicoes original

    # treina com tudo
    python treinar_cnn.py

    # gerador nunca visto: tira "FLUX" do treino e usa só no teste
    python treinar_cnn.py --fontes_teste FLUX

    # recorte de 256px (se as redes sociais reduzirem muito as imagens)
    python treinar_cnn.py --tamanho 256

    # trabalhar com um pedaço do dataset (ex.: 2000 imagens por gerador/base)
    python treinar_cnn.py --max_grupos 2000

Dependências: pip install torch numpy pillow
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from comum import (balancear, dividir_por_grupo, ela_de_arquivo, ler_index,
                   limitar_grupos, metricas)


# ---------------------------------------------------------------- dados
class ELADataset(Dataset):
    """Calcula o ELA na hora (nos processos do DataLoader), sem cache em disco."""

    def __init__(self, linhas, tamanho, qualidade, aumentar=False):
        self.linhas, self.tamanho, self.qualidade = linhas, tamanho, qualidade
        self.aumentar = aumentar

    def __len__(self):
        return len(self.linhas)

    def __getitem__(self, i):
        l = self.linhas[i]
        ela = ela_de_arquivo(l["caminho"], self.tamanho, self.qualidade)
        x = torch.from_numpy(ela.astype(np.float32)).permute(2, 0, 1) / 255.0
        if self.aumentar and random.random() < 0.5:
            x = torch.flip(x, dims=[2])          # espelhar não altera o padrão ELA
        return x, l["rotulo"]


def criar_loader(linhas, args, treino=False):
    return DataLoader(
        ELADataset(linhas, args.tamanho, args.qualidade, aumentar=treino),
        batch_size=args.lote, shuffle=treino, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0)


# ---------------------------------------------------------------- modelo
class CNNArtigo(nn.Module):
    """Figura 4c do artigo. Com entrada de 512px é idêntica à original;
    para outros tamanhos, o número de entradas da camada final é
    calculado automaticamente."""

    def __init__(self, tamanho=512, canais=3):
        super().__init__()

        def bloco(c_in, c_out, k):
            return [nn.Conv2d(c_in, c_out, k, padding="same"),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            nn.BatchNorm2d(canais),            # normaliza a escala do ELA bruto
            *bloco(canais, 4, 5),              # Stage 1
            nn.MaxPool2d(5, stride=4),
            *bloco(4, 8, 5),                   # Stage 2
            nn.MaxPool2d(4, stride=4),
            *bloco(8, 16, 5),                  # Stage 3
            nn.MaxPool2d(2, stride=4),
            *bloco(16, 32, 2),                 # Stage 4
        )
        with torch.no_grad():
            self.features.eval()
            n = self.features(torch.zeros(1, canais, tamanho, tamanho)).numel()
            self.features.train()
        self.classificador = nn.Linear(n, 2)

    def forward(self, x):
        return self.classificador(torch.flatten(self.features(x), 1))


@torch.no_grad()
def prever(modelo, loader, disp):
    modelo.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += modelo(x.to(disp, non_blocking=True)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return ys, ps


def resumo(nome, linhas):
    c = Counter((l["condicao"], "ia" if l["rotulo"] else "real") for l in linhas)
    partes = ", ".join(f"{cond}/{r}={n}" for (cond, r), n in sorted(c.items()))
    print(f"  {nome:7s} {len(linhas):7d}  [{partes}]")


# ---------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--index", default="index.csv")
    a.add_argument("--tamanho", type=int, default=512)
    a.add_argument("--qualidade", type=int, default=95, help="qualidade JPEG do ELA")
    a.add_argument("--treino_condicoes", nargs="*", default=None,
                   help="condições usadas em treino/validação (padrão: todas)")
    a.add_argument("--fontes_teste", nargs="*", default=[],
                   help="fontes (geradores/bases) usadas SÓ no teste")
    a.add_argument("--max_grupos", type=int, default=0,
                   help="máx. de imagens por (rótulo, fonte); 0 = todas")
    a.add_argument("--epocas", type=int, default=100)
    a.add_argument("--lote", type=int, default=16)
    a.add_argument("--lr", type=float, default=0.01)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--saida", default="resultados")
    a.add_argument("--nome", default=None, help="nome do experimento (pasta de saída)")
    args = a.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {disp}")

    # ---- leitura e filtros
    linhas = ler_index(args.index)
    pequenas = [l for l in linhas if min(l["largura"], l["altura"]) < args.tamanho]
    if pequenas:
        c = Counter((l["condicao"], "ia" if l["rotulo"] else "real") for l in pequenas)
        print(f"[AVISO] {len(pequenas)} imagens menores que {args.tamanho}px descartadas: {dict(c)}")
    linhas = [l for l in linhas if min(l["largura"], l["altura"]) >= args.tamanho]
    linhas = limitar_grupos(linhas, args.max_grupos, args.seed)

    # ---- fontes reservadas para o teste (gerador nunca visto)
    reservadas = [l for l in linhas if l["fonte"] in args.fontes_teste]
    linhas = [l for l in linhas if l["fonte"] not in args.fontes_teste]

    splits = dividir_por_grupo(linhas, seed=args.seed)
    conds = set(args.treino_condicoes) if args.treino_condicoes else None

    def filtra(ls):
        return ls if conds is None else [l for l in ls if l["condicao"] in conds]

    treino = balancear(filtra(splits["treino"]), args.seed)
    val = balancear(filtra(splits["val"]), args.seed)
    teste = balancear(splits["teste"] + reservadas, args.seed)   # todas as condições

    print("Conjuntos (após balancear):")
    resumo("treino", treino); resumo("val", val); resumo("teste", teste)
    if not treino or not val:
        print("Treino ou validação vazio. Confira o index e os filtros.")
        return

    # ---- treino
    nome = args.nome or "treino_" + ("todas" if conds is None else "+".join(sorted(conds)))
    if args.fontes_teste:
        nome += "__sem_" + "+".join(args.fontes_teste)
    saida = Path(args.saida) / nome
    saida.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(saida / "args.json", "w"), indent=2, ensure_ascii=False)

    dl_treino, dl_val = criar_loader(treino, args, treino=True), criar_loader(val, args)
    modelo = CNNArtigo(args.tamanho).to(disp)
    otim = torch.optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9)   # SGDM, como no artigo
    perda_fn = nn.CrossEntropyLoss()

    melhor_acc, historico = -1.0, []
    for ep in range(1, args.epocas + 1):
        modelo.train()
        soma, n = 0.0, 0
        for x, y in dl_treino:
            x, y = x.to(disp, non_blocking=True), y.to(disp, non_blocking=True)
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

    # ---- teste: por condição e por fonte
    modelo.load_state_dict(torch.load(saida / "melhor_modelo.pt", map_location=disp))
    ys, ps = prever(modelo, criar_loader(teste, args), disp)
    for l, y_pred in zip(teste, ps):
        l["pred"] = y_pred

    resultados = {"melhor_acc_val": melhor_acc, "geral": metricas(ys, ps),
                  "por_condicao": {}, "por_fonte": {}}
    print("\n=== TESTE POR CONDIÇÃO ===")
    for cond in sorted({l["condicao"] for l in teste}):
        sub = [l for l in teste if l["condicao"] == cond]
        m = metricas([l["rotulo"] for l in sub], [l["pred"] for l in sub])
        resultados["por_condicao"][cond] = m
        print(f"  {cond:10s} n={m['n']:6d}  acc {m['acuracia']:.3f}  P {m['precisao']:.3f}  "
              f"R {m['recall']:.3f}  F1 {m['f1']:.3f}  [[TP FN][FP TN]]={m['matriz_TP_FN_FP_TN']}")

    print("\n=== TESTE POR FONTE x CONDIÇÃO (acerto) ===")
    print("  (cada fonte tem uma classe só: para IA o acerto é o recall, para reais é a especificidade)")
    for fonte in sorted({l["fonte"] for l in teste}):
        resultados["por_fonte"][fonte] = {}
        for cond in sorted({l["condicao"] for l in teste if l["fonte"] == fonte}):
            sub = [l for l in teste if l["fonte"] == fonte and l["condicao"] == cond]
            acc = sum(l["pred"] == l["rotulo"] for l in sub) / len(sub)
            resultados["por_fonte"][fonte][cond] = {"n": len(sub), "acerto": acc}
            marca = "  (nunca visto no treino)" if fonte in args.fontes_teste else ""
            print(f"  {fonte:15s} {cond:10s} n={len(sub):6d}  acerto {acc:.3f}{marca}")

    json.dump(resultados, open(saida / "metricas.json", "w"), indent=2, ensure_ascii=False)
    json.dump(historico, open(saida / "historico.json", "w"), indent=2)
    print(f"\nResultados salvos em {saida}/")


if __name__ == "__main__":
    main()
