from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import os


def classify_ela_image(image):
    """
    Classifica uma imagem ELA diretamente.

    Args:
        image (np.array): imagem ELA (BGR - OpenCV)

    Returns:
        dict com classificação e métricas
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    total_pixels = image.shape[0] * image.shape[1]

    # --- métricas ---
    gray_mask = (S < 40)
    color_mask = (S >= 40)
    black_mask = (V < 30)

    gray_ratio = np.sum(gray_mask) / total_pixels
    color_ratio = np.sum(color_mask) / total_pixels
    black_ratio = np.sum(black_mask) / total_pixels

    # --- classificação ---
    result = {
        "class": None,
        "confidence": 0.0,
        "gray_ratio": gray_ratio,
        "color_ratio": color_ratio,
        "black_ratio": black_ratio,
    }

    # 1. recompressão (WhatsApp)
    if black_ratio > 0.85:
        result["class"] = "RECOMPRESSED (WhatsApp / Social Media)"
        result["confidence"] = black_ratio
        return result

    # 2. IA
    if color_ratio > 0.25:
        result["class"] = "LIKELY AI-GENERATED"
        result["confidence"] = color_ratio
        return result

    # 3. real
    if color_ratio < 0.15 and black_ratio > 0.6:
        result["class"] = "LIKELY REAL (CAMERA)"
        result["confidence"] = black_ratio
        return result

    # 4. incerto
    result["class"] = "UNCERTAIN"
    result["confidence"] = 0.5

    return result


def resize_keep_aspect(img, max_w=800, max_h=600):
    h, w = img.shape[:2]

    scale = min(max_w / w, max_h / h)
    if scale >= 1:
        return img  # não aumenta imagem

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


def count_gray_and_color_pixels(image):
    """
    Conta pixels cinza/branco vs coloridos.

    Args:
        image (np.array): imagem BGR (OpenCV)

    Returns:
        dict
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    total_pixels = image.shape[0] * image.shape[1]

    # --- cinza/branco (baixa saturação) ---
    gray_mask = (S < 40)

    # opcional: separar branco de preto (se quiser depois)
    # white_mask = (S < 40) & (V > 200)
    black_mask = (S < 40) & (V < 50)

    black_count = np.sum(black_mask)

    gray_count = np.sum(gray_mask)

    # --- colorido ---
    color_mask = (S >= 40)
    color_count = np.sum(color_mask)



    return {
        "gray_ratio": gray_count / total_pixels,
        "color_ratio": color_count / total_pixels,
        "black_ratio": black_count / total_pixels,
        "gray_count": int(gray_count),
        "color_count": int(color_count),
        "black_count": int(black_count)
    }

def compute_ela(image_path, quality=95):
    """
    Aplica Error Level Analysis (ELA) em uma imagem.

    Args:
        image_path (str): caminho da imagem original
        quality (int): qualidade JPEG para recompressão

    Returns:
        np.array: imagem ELA em formato OpenCV (BGR)
    """

    # Abre imagem
    original = Image.open(image_path).convert('RGB')

    # Salva temporariamente com compressão JPEG
    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)

    # Reabre imagem comprimida
    compressed = Image.open(temp_path)

    # Calcula diferença
    diff = ImageChops.difference(original, compressed)

    # Amplifica diferença (ESSENCIAL)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    # Converte para OpenCV (BGR)
    ela_image = cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR)

    # Remove arquivo temporário
    os.remove(temp_path)

    return ela_image

ela_real = compute_ela("gustavo_real.jpg")
ela_real = cv2.rotate(ela_real, cv2.ROTATE_90_CLOCKWISE)

ela = compute_ela("gustavo_ia_gemini.jpg")


ela_zap = compute_ela("gustavo_zap.jpg")

ela_ia_zap = compute_ela("gustavo_ia_gemini_zap.jpg")

cv2.namedWindow("ELA", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ELA", 800, 600)

cv2.namedWindow("ELA_real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ELA_real", 800, 600)

cv2.namedWindow("ELA_zap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ELA_zap", 800, 600)


cv2.namedWindow("ELA_ia_zap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ELA_ia_zap", 800, 600)

cv2.imshow("ELA", ela)
cv2.imshow("ELA_real", ela_real)
cv2.imshow("ELA_zap", ela_zap)
cv2.imshow("ELA_ia_zap", ela_ia_zap)

print("\n===== ANALISE ELA =====\n")

images = {
    "REAL (camera)": ela_real,
    "IA (gemini)": ela,
    "WHATSAPP (real comprimida)": ela_zap,
    "IA via WHATSAPP": ela_ia_zap,
}

for name, img in images.items():

    color_analysis = count_gray_and_color_pixels(img)

    print(f"--- {name} ---")

    print(f"Gray ratio: {color_analysis['gray_ratio']:.4f}")
    print(f"Color ratio: {color_analysis['color_ratio']:.4f}")
    print(f"Black ratio: {color_analysis['black_ratio']:.4f}")
    print(f"gray_count: {color_analysis['gray_count']:.4f}")
    print(f"color_count: {color_analysis['color_count']:.4f}")
    print(f"black_count: {color_analysis['black_count']:.4f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()


