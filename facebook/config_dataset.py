"""
CONFIGURAÇÃO DO DATASET - preencha depois de baixar o TrueFake.

Cada entrada de FONTES descreve UMA pasta (lida recursivamente) e diz:
  rotulo   -> "real" ou "ia"
  condicao -> "original", "facebook", "x", "telegram", "whatsapp", ...
  fonte    -> de onde veio a imagem: o gerador (ex: "SD2.1", "FLUX") ou a base
              de fotos reais (ex: "FFHQ", "FORLAB"). Se for None, usa o nome
              da pasta encontrada (útil com curingas, ver exemplo abaixo).

Para a divisão sem vazamento funcionar, a MESMA imagem precisa ter a mesma
"fonte" e o mesmo identificador de grupo (nome do arquivo, ou o que
GRUPO_REGEX extrair dele) em todas as condições.

O caminho aceita curingas (*). Exemplo, supondo uma estrutura como
    TrueFake/PreSocial/Fake/SD2.1/...
    TrueFake/Facebook/Fake/SD2.1/...
a entrada
    {"pasta": r"D:/TrueFake/Facebook/Fake/*", "rotulo": "ia",
     "condicao": "facebook", "fonte": None}
cria uma fonte para cada subpasta (SD2.1, FLUX, ...) automaticamente.

DICA: para deixar os rostos de GAN de fora (só rostos -> viés de conteúdo),
basta não listar as pastas deles aqui.
"""

FONTES = [
    # ------------------------------------------------------------ REAIS (Facebook)
    # "*" -> cada subpasta vira uma fonte: FFHQ e FORLAB
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Real/*", "rotulo": "real", "condicao": "facebook", "fonte": None},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Real/*", "rotulo": "real", "condicao": "telegram", "fonte": None},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Real/*", "rotulo": "real", "condicao": "twitter", "fonte": None},

    # ------------------------------------------------------------ IA - difusão (Facebook)
    # as subpastas animals/faces/general/landscapes são lidas automaticamente
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/FLUX.1",             "rotulo": "ia", "condicao": "facebook", "fonte": "FLUX.1"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusion1.5", "rotulo": "ia", "condicao": "facebook", "fonte": "SD1.5"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusion2",   "rotulo": "ia", "condicao": "facebook", "fonte": "SD2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusion3",   "rotulo": "ia", "condicao": "facebook", "fonte": "SD3"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusionXL",  "rotulo": "ia", "condicao": "facebook", "fonte": "SDXL"},

    # ------------------------------------------------------------ IA - GANs (só rostos)
    # Deixadas de fora de propósito: só têm rostos, o que cria viés de conteúdo.
    # Descomente se quiserem um experimento específico com elas.
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StyleGAN",  "rotulo": "ia", "condicao": "facebook", "fonte": "StyleGAN"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StyleGAN2", "rotulo": "ia", "condicao": "facebook", "fonte": "StyleGAN2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StyleGAN3", "rotulo": "ia", "condicao": "facebook", "fonte": "StyleGAN3"},

    # ------------------------------------------------------------ outras condições (depois)
    # Quando baixarem as originais (pré-social), X e Telegram, adicionem entradas
    # iguais às de cima trocando o caminho e a "condicao" ("original", "x", "telegram").
    # Use os MESMOS nomes de "fonte" (ex: "SD1.5") para o pareamento funcionar.

    # ------------------------------------------------------------ IA - difusão (Facebook)
    # as subpastas animals/faces/general/landscapes são lidas automaticamente
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/FLUX.1", "rotulo": "ia", "condicao": "facebook", "fonte": "FLUX.1"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusion1.5", "rotulo": "ia",
     "condicao": "facebook", "fonte": "SD1.5"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusion2", "rotulo": "ia", "condicao": "facebook", "fonte": "SD2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusion3", "rotulo": "ia", "condicao": "facebook", "fonte": "SD3"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StableDiffusionXL", "rotulo": "ia", "condicao": "facebook", "fonte": "SDXL"},

    # ------------------------------------------------------------ IA - GANs (só rostos)
    # Deixadas de fora de propósito: só têm rostos, o que cria viés de conteúdo.
    # Descomente se quiserem um experimento específico com elas.
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StyleGAN", "rotulo": "ia", "condicao": "facebook", "fonte": "StyleGAN"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StyleGAN2", "rotulo": "ia", "condicao": "facebook", "fonte": "StyleGAN2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Facebook/Fake/StyleGAN3", "rotulo": "ia", "condicao": "facebook", "fonte": "StyleGAN3"},

    # ------------------------------------------------------------ IA - difusão (Telegram)
    # as subpastas animals/faces/general/landscapes são lidas automaticamente
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/FLUX.1", "rotulo": "ia", "condicao": "telegram", "fonte": "FLUX.1"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StableDiffusion1.5", "rotulo": "ia", "condicao": "telegram", "fonte": "SD1.5"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StableDiffusion2", "rotulo": "ia", "condicao": "telegram", "fonte": "SD2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StableDiffusion3", "rotulo": "ia", "condicao": "telegram", "fonte": "SD3"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StableDiffusionXL", "rotulo": "ia", "condicao": "telegram", "fonte": "SDXL"},

    # ------------------------------------------------------------ IA - GANs (só rostos)
    # Deixadas de fora de propósito: só têm rostos, o que cria viés de conteúdo.
    # Descomente se quiserem um experimento específico com elas.
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StyleGAN", "rotulo": "ia", "condicao": "telegram", "fonte": "StyleGAN"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StyleGAN2", "rotulo": "ia", "condicao": "telegram", "fonte": "StyleGAN2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Telegram/Fake/StyleGAN3", "rotulo": "ia", "condicao": "telegram", "fonte": "StyleGAN3"},

    # ------------------------------------------------------------ IA - difusão (Twitter)
    # as subpastas animals/faces/general/landscapes são lidas automaticamente
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/FLUX.1", "rotulo": "ia", "condicao": "twitter", "fonte": "FLUX.1"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StableDiffusion1.5", "rotulo": "ia",
     "condicao": "twitter", "fonte": "SD1.5"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StableDiffusion2", "rotulo": "ia", "condicao": "twitter", "fonte": "SD2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StableDiffusion3", "rotulo": "ia", "condicao": "twitter", "fonte": "SD3"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StableDiffusionXL", "rotulo": "ia", "condicao": "twitter", "fonte": "SDXL"},

    # ------------------------------------------------------------ IA - GANs (só rostos)
    # Deixadas de fora de propósito: só têm rostos, o que cria viés de conteúdo.
    # Descomente se quiserem um experimento específico com elas.
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StyleGAN", "rotulo": "ia", "condicao": "twitter", "fonte": "StyleGAN"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StyleGAN2", "rotulo": "ia", "condicao": "twitter", "fonte": "StyleGAN2"},
    {"pasta": r"C:/Users/Gustavo/Desktop/codes/Twitter/Fake/StyleGAN3", "rotulo": "ia", "condicao": "twitter", "fonte": "StyleGAN3"},

]

# Expressão regular para extrair o identificador do grupo do NOME do arquivo
# (sem extensão). None = usa o nome inteiro.
# Ex.: se o original é "img_0042.png" e o compartilhado é "img_0042_fb.jpg",
# use r"(img_\d+)". O primeiro grupo de captura vira o identificador.
GRUPO_REGEX = None

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
