"""
Classifica imagens novas com um modelo já treinado por treinar_cnn.py.

Uso:
    python classificar.py foto.jpg --modelo ela_small_image
    python classificar.py foto1.jpg foto2.png pasta_com_imagens/ --modelo ela_big_image/treino_todas
    python classificar.py foto.jpg                      # usa a pasta 'resultados'
    python classificar.py foto.jpg --salvar_ela         # salva o mapa ELA para visualizar

O tamanho do recorte e a qualidade do ELA são lidos do args.json salvo no
treino. O próprio args.json também informa se o modelo foi treinado com ou
sem ELA ("usar_ela"), então não é preciso passar nada na classificação.

Sem --modelo, é usada a pasta 'resultados' (com aviso). O valor aceita a
pasta do experimento, a pasta-base (resultados, ela_big_image, ...) ou só o
nome da pasta-base.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comum import imagem_de_arquivo
from treinar_cnn import CNNArtigo
import statistics

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
RAIZ = Path(__file__).resolve().parent


def resolver_experimento(valor):
    """Aceita a pasta do experimento, a pasta-base ou só o nome da pasta-base."""
    p = Path(valor)
    if not p.is_absolute() and not p.exists():
        p = RAIZ / valor

    if (p / "args.json").is_file() and (p / "melhor_modelo.pt").is_file():
        return p

    if p.is_dir():
        candidatos = sorted(
            d for d in p.iterdir()
            if (d / "args.json").is_file() and (d / "melhor_modelo.pt").is_file()
        )
        if len(candidatos) == 1:
            print(f"[INFO] usando experimento: {candidatos[0]}")
            return candidatos[0]
        if not candidatos:
            raise SystemExit(
                f"[ERRO] nenhum experimento (args.json + melhor_modelo.pt) em {p}")
        nomes = ", ".join(d.name for d in candidatos)
        raise SystemExit(
            f"[ERRO] vários experimentos em {p}: {nomes}. "
            f"Use --modelo {p.name}/<nome>")

    raise SystemExit(f"[ERRO] modelo não encontrado: {valor}")


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
    ap.add_argument("--modelo", default=None,
                    help="experimento, pasta-base ou nome da pasta-base "
                         "(padrão: resultados)")
    ap.add_argument("--limiar", type=float, default=0.5,
                    help="probabilidade de IA a partir da qual a imagem é classificada como IA")
    ap.add_argument("--salvar_ela", action="store_true",
                    help="salva o mapa ELA (amplificado) ao lado de cada imagem")
    args = ap.parse_args()

    if args.modelo is None:
        print("[AVISO] --modelo não informado; usando a pasta 'resultados'")
        args.modelo = "resultados"
    pasta = resolver_experimento(args.modelo)
    cfg = json.load(open(pasta / "args.json", encoding="utf-8"))
    tamanho, qualidade = cfg["tamanho"], cfg["qualidade"]
    usar_ela = cfg.get("usar_ela", True)   # modelos antigos não têm o campo

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = CNNArtigo(tamanho).to(disp)
    modelo.load_state_dict(torch.load(pasta / "melhor_modelo.pt", map_location=disp))
    modelo.eval()

    # --- Configurações de exibição de resumo ---
    # Você pode adicionar essas opções no argparse (ex: parser.add_argument('--apenas-resumo', action='store_true'))
    apenas_resumo = getattr(args, "apenas_resumo", True)  # Se True, oculta o print por arquivo
    intervalo_ponto = getattr(args, "intervalo_ponto", 10)  # Printa 1 ponto a cada X arquivos

    # --- Contadores e armazenamento de estatísticas ---
    total_arquivos = 0
    count_ia = 0
    count_real = 0
    count_nao_classificados = 0

    probs_ia = []  # Guardará as % das classificadas como IA
    probs_real = []  # Guardará as % das classificadas como REAL
    probs_todas = []  # Guardará todas as % válidas

    modo = f"ELA (qualidade {qualidade})" if usar_ela else "RGB (sem ELA)"
    print(f"modelo: {pasta}  (recorte {tamanho}px, entrada: {modo})\n")
    if args.salvar_ela and not usar_ela:
        print("[AVISO] modelo treinado sem ELA; --salvar_ela será ignorado\n")
    if apenas_resumo:
        print("Processando imagens", end="", flush=True)

    for arq in listar_imagens(args.imagens):
        total_arquivos += 1

        # Se a opção 'apenas_resumo' estiver ativa, imprime '.' a cada X arquivos
        if apenas_resumo and total_arquivos % intervalo_ponto == 0:
            print(".", end="", flush=True)

        try:
            formato = Image.open(arq).format
            entrada = imagem_de_arquivo(arq, tamanho, qualidade, usar_ela)
        except Exception as e:
            count_nao_classificados += 1
            if not apenas_resumo:
                print(f"{arq.name}: erro ao abrir ({e})")
            continue

        if entrada is None:
            count_nao_classificados += 1
            if not apenas_resumo:
                print(f"{arq.name}: imagem menor que {tamanho}px, não dá para classificar")
            continue

        x = torch.from_numpy(entrada.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            prob_ia = torch.softmax(modelo(x.to(disp)), dim=1)[0, 1].item()

        # Converter para porcentagem arredondada a 1 casa decimal (ideal para calcular a moda)
        pct_ia = round(prob_ia * 100, 1)
        probs_todas.append(pct_ia)

        classe = "IA" if prob_ia >= args.limiar else "REAL"

        if classe == "IA":
            count_ia += 1
            probs_ia.append(pct_ia)
        else:
            count_real += 1
            probs_real.append(pct_ia)

        if not apenas_resumo:
            aviso = "  [atenção: arquivo não é JPEG]" if formato != "JPEG" else ""
            print(f"{arq.name}: {classe}  (probabilidade de IA: {prob_ia:.1%}){aviso}")

        if args.salvar_ela and usar_ela:
            vis = np.clip(entrada.astype(np.float32) * 10, 0, 255).astype(np.uint8)
            destino = arq.with_name(f"{arq.stem}_ela.png")
            Image.fromarray(vis).save(destino)

    if apenas_resumo:
        print(" Concluído!\n")

    # --- Função para calcular Média, Mediana e Moda ---
    def calcular_estatisticas(lista_pct):
        if not lista_pct:
            return None
        media = statistics.mean(lista_pct)
        mediana = statistics.median(lista_pct)
        modas = statistics.multimode(lista_pct)
        return media, mediana, modas

    def formatar_estatisticas(estat):
        if not estat:
            return "Sem dados"
        media, mediana, modas = estat
        modas_str = ", ".join(f"{m}%" for m in modas[:5])  # Exibe até 5 modas se houver empate
        if len(modas) > 5:
            modas_str += "..."
        return f"Média = {media:.2f}% | Mediana = {mediana:.2f}% | Moda = {modas_str}"

    # --- Relatório Final ---
    print("=" * 60)
    print("              RESUMO FINAL DA CLASSIFICAÇÃO             ")
    print("=" * 60)
    print(f"Total de arquivos analisados: {total_arquivos}")
    print(f"  • Classificadas como IA:   {count_ia} ({count_ia / total_arquivos:.1%})")
    print(f"  • Classificadas como REAL: {count_real} ({count_real / total_arquivos:.1%})")
    print(f"  • Não classificadas:       {count_nao_classificados} ({count_nao_classificados / total_arquivos:.1%})")
    print("-" * 60)

    print("Estatísticas das probabilidades de ser IA (%):")
    print(f"  - Imagens REAL (Erros):  {formatar_estatisticas(calcular_estatisticas(probs_real))}")
    print(f"  - Imagens IA (Acertos):  {formatar_estatisticas(calcular_estatisticas(probs_ia))}")
    print(f"  - Geral (Todas válidas): {formatar_estatisticas(calcular_estatisticas(probs_todas))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
