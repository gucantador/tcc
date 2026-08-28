from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import os
import shutil


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
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


def count_gray_and_color_pixels(image):
    """
    Conta pixels cinza/branco vs coloridos.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    total_pixels = image.shape[0] * image.shape[1]

    gray_mask = (S < 40)
    black_mask = (S < 40) & (V < 50)

    black_count = np.sum(black_mask)
    gray_count = np.sum(gray_mask)

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
    """

    original = Image.open(image_path).convert('RGB')

    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)

    compressed = Image.open(temp_path)

    diff = ImageChops.difference(original, compressed)

    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    ela_image = cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR)

    os.remove(temp_path)

    return ela_image


def save_output(output_dir, original_path, ela_image, result):
    """
    Salva:
    - imagem original
    - imagem ELA
    - relatório TXT
    """

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_path))[0]

    # --- copia original ---
    original_output = os.path.join(
        output_dir,
        f"{base_name}_original.jpg"
    )

    shutil.copy2(original_path, original_output)

    # --- salva ELA ---
    ela_output = os.path.join(
        output_dir,
        f"{base_name}_ELA.jpg"
    )

    cv2.imwrite(ela_output, ela_image)

    # --- salva relatório ---
    report_output = os.path.join(
        output_dir,
        f"{base_name}_report.txt"
    )

    with open(report_output, "w", encoding="utf-8") as f:

        f.write(f"Imagem: {base_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"CLASS: {result['class']}\n")
        f.write(f"CONFIDENCE: {result['confidence']:.4f}\n\n")

        f.write(f"Gray ratio: {result['gray_ratio']:.4f}\n")
        f.write(f"Color ratio: {result['color_ratio']:.4f}\n")
        f.write(f"Black ratio: {result['black_ratio']:.4f}\n")

    print(f"[OK] Output salvo em: {output_dir}")


# =========================================================
# LISTA DE IMAGENS
# =========================================================

image_paths = [
    "gustavo_real.jpg",
    "gustavo_ia_gemini.jpg",
    "gustavo_zap.jpg",
    "gustavo_ia_gemini_zap.jpg"
]

output_root = "output_ela"

print("\n===== ANALISE ELA =====\n")

for image_path in image_paths:

    print(f"\nProcessando: {image_path}")

    ela_img = compute_ela(image_path)

    # rotação opcional
    if "real" in image_path.lower():
        ela_img = cv2.rotate(ela_img, cv2.ROTATE_90_CLOCKWISE)

    metrics = count_gray_and_color_pixels(ela_img)
    classification = classify_ela_image(ela_img)

    result = {
        **metrics,
        **classification
    }

    print(f"Classificação: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")

    print(f"Gray ratio: {result['gray_ratio']:.4f}")
    print(f"Color ratio: {result['color_ratio']:.4f}")
    print(f"Black ratio: {result['black_ratio']:.4f}")

    # salva tudo
    save_output(
        output_dir=output_root,
        original_path=image_path,
        ela_image=ela_img,
        result=result
    )

print("\nFINALIZADO.")