"""Funções compartilhadas: leitura do índice, ELA, divisão, balanceamento e métricas."""

import csv
import io
import random
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------- índice
def ler_index(caminho):
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for l in linhas:
        l["rotulo"] = int(l["rotulo"])
        l["largura"] = int(l["largura"])
        l["altura"] = int(l["altura"])
    return linhas


def chave_grupo(l):
    """A mesma imagem em todas as condições compartilha esta chave."""
    return (l["rotulo"], l["fonte"], l["grupo"])


# ---------------------------------------------------------------- ELA
def carregar_rgb(caminho):
    img = Image.open(caminho)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fundo = Image.new("RGB", img.size, (255, 255, 255))
        fundo.paste(img, mask=img.split()[-1])
        return fundo
    return img.convert("RGB")


def recorte_central_alinhado(img, tamanho, alinhamento=16):
    """Recorte central com o canto superior esquerdo em múltiplo de 16.
    Assim os blocos 8x8 (e os macroblocos 16x16 do croma) do JPEG original
    continuam alinhados com os da recompressão do ELA, e dá para calcular o
    ELA só no recorte em vez da imagem inteira (muito mais rápido)."""
    w, h = img.size
    if min(w, h) < tamanho:
        return None
    esq = ((w - tamanho) // 2) // alinhamento * alinhamento
    topo = ((h - tamanho) // 2) // alinhamento * alinhamento
    return img.crop((esq, topo, esq + tamanho, topo + tamanho))


def calcular_ela(img, qualidade=95):
    """|img - JPEG(img, qualidade)| sem normalização por imagem."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=qualidade)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    a = np.asarray(img, dtype=np.int16)
    b = np.asarray(rec, dtype=np.int16)
    return np.abs(a - b).astype(np.uint8)


def ela_de_arquivo(caminho, tamanho=512, qualidade=95):
    img = recorte_central_alinhado(carregar_rgb(caminho), tamanho)
    if img is None:
        return None
    return calcular_ela(img, qualidade)


# ---------------------------------------------------------------- divisão
def limitar_grupos(linhas, max_grupos, seed):
    """Mantém no máximo max_grupos imagens (com todas as suas versões)
    por (rótulo, fonte). Útil para trabalhar com um recorte do TrueFake."""
    if not max_grupos:
        return linhas
    grupos = defaultdict(set)
    for l in linhas:
        grupos[(l["rotulo"], l["fonte"])].add(chave_grupo(l))
    rng = random.Random(seed)
    manter = set()
    for _, gs in sorted(grupos.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        manter.update(gs[:max_grupos])
    return [l for l in linhas if chave_grupo(l) in manter]


def dividir_por_grupo(linhas, fr_val=0.15, fr_teste=0.15, seed=42):
    """Divide por grupo, estratificado por (rótulo, fonte): todas as versões
    de uma mesma imagem caem no mesmo conjunto (sem vazamento) e cada gerador
    aparece em proporção parecida em treino, validação e teste."""
    grupos = defaultdict(set)
    for l in linhas:
        grupos[(l["rotulo"], l["fonte"])].add(chave_grupo(l))
    rng = random.Random(seed)
    destino = {}
    for _, gs in sorted(grupos.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        n = len(gs)
        n_te, n_va = round(n * fr_teste), round(n * fr_val)
        for i, g in enumerate(gs):
            destino[g] = "teste" if i < n_te else "val" if i < n_te + n_va else "treino"
    splits = defaultdict(list)
    for l in linhas:
        splits[destino[chave_grupo(l)]].append(l)
    return splits


def balancear(linhas, seed):
    """Dentro de cada condição, reduz a classe maior ao tamanho da menor."""
    rng = random.Random(seed)
    por_cond = defaultdict(lambda: {0: [], 1: []})
    for l in linhas:
        por_cond[l["condicao"]][l["rotulo"]].append(l)
    saida = []
    for cond in sorted(por_cond):
        reais, ias = por_cond[cond][0], por_cond[cond][1]
        n = min(len(reais), len(ias))
        if n == 0:
            print(f"[AVISO] condição '{cond}' tem só uma classe; ficou de fora do balanceamento")
            continue
        saida += rng.sample(reais, n) + rng.sample(ias, n)
    return saida


# ---------------------------------------------------------------- métricas
def metricas(y_true, y_pred):
    """Positivo = IA (rótulo 1), como no artigo base."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
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
            "matriz_TP_FN_FP_TN": [[tp, fn], [fp, tn]]}
