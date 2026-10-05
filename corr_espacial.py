import cv2
import numpy as np

# =========================
# 📸 Carregar imagem
# =========================
def carregar_imagem(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


# =========================
# 📐 1. Correlação espacial
# =========================
def correlacao_espacial(canal):
    # deslocamento horizontal
    h_corr = np.corrcoef(canal[:, :-1].flatten(), canal[:, 1:].flatten())[0,1]
    
    # deslocamento vertical
    v_corr = np.corrcoef(canal[:-1, :].flatten(), canal[1:, :].flatten())[0,1]
    
    return h_corr, v_corr


# =========================
# 🌈 2. Correlação entre canais
# =========================
def correlacao_canais(img):
    R = img[:,:,0].flatten()
    G = img[:,:,1].flatten()
    B = img[:,:,2].flatten()
    
    corr_RG = np.corrcoef(R, G)[0,1]
    corr_GB = np.corrcoef(G, B)[0,1]
    corr_RB = np.corrcoef(R, B)[0,1]
    
    return corr_RG, corr_GB, corr_RB


# =========================
# 🔲 3. Padrão 2x2 (CFA residual)
# =========================
def padrao_2x2(img):
    H, W, _ = img.shape
    
    grupos = {
        "00": [],
        "01": [],
        "10": [],
        "11": []
    }
    
    for i in range(H):
        for j in range(W):
            chave = f"{i%2}{j%2}"
            grupos[chave].append(img[i,j])
    
    medias = []
    for g in grupos.values():
        medias.append(np.mean(g))
    
    # variância entre grupos → indica padrão
    return np.var(medias)


# =========================
# 🚀 Extrair todas features
# =========================
def extrair_features(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    # 1. Correlação espacial
    Rh, Rv = correlacao_espacial(R)
    Gh, Gv = correlacao_espacial(G)
    Bh, Bv = correlacao_espacial(B)
    
    # 2. Correlação entre canais
    corr_RG, corr_GB, corr_RB = correlacao_canais(img)
    
    # 3. Padrão 2x2
    cfa_score = padrao_2x2(img)
    
    features = [
        Rh, Rv,
        Gh, Gv,
        Bh, Bv,
        corr_RG, corr_GB, corr_RB,
        cfa_score
    ]
    
    return np.array(features)


# =========================
# 🧪 Teste
# =========================
if __name__ == "__main__":
    img = carregar_imagem("teste_cortina_real.jpg")
    
    features = extrair_features(img)
    
    print("Features extraídas:")
    print(features)