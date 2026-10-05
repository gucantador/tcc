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
        result["class"] = "RECOMPRESSED"
        result["confidence"] = black_ratio
        return result

    # 2. IA
    if color_ratio > 0.25:
        result["class"] = "AI"
        result["confidence"] = color_ratio
        return result

    # 3. real
    if color_ratio < 0.15 and black_ratio > 0.6:
        result["class"] = "REAL"
        result["confidence"] = black_ratio
        return result

    # 4. incerto
    result["class"] = "UNCERTAIN"
    result["confidence"] = 0.5

    return result


def resize_keep_aspect(img, max_w=800, max_h=600):

    h, w = img.shape[:2]

    scale = min(
        max_w / w,
        max_h / h
    )

    if scale >= 1:
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        img,
        (new_w, new_h)
    )


def count_gray_and_color_pixels(image):
    """
    Conta pixels cinza/branco vs coloridos.

    Args:
        image (np.array): imagem BGR (OpenCV)

    Returns:
        dict
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    H, S, V = (
        hsv[:, :, 0],
        hsv[:, :, 1],
        hsv[:, :, 2]
    )

    total_pixels = (
        image.shape[0] *
        image.shape[1]
    )

    # --- cinza/branco ---
    gray_mask = (S < 40)

    # --- preto ---
    black_mask = (
        (S < 40) &
        (V < 50)
    )

    black_count = np.sum(
        black_mask
    )

    gray_count = np.sum(
        gray_mask
    )

    # --- colorido ---
    color_mask = (S >= 40)

    color_count = np.sum(
        color_mask
    )

    return {
        "gray_ratio":
            gray_count / total_pixels,

        "color_ratio":
            color_count / total_pixels,

        "black_ratio":
            black_count / total_pixels,

        "gray_count":
            int(gray_count),

        "color_count":
            int(color_count),

        "black_count":
            int(black_count)
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
    original = Image.open(
        image_path
    ).convert('RGB')

    # Caminho temporário
    temp_path = "temp_ela.jpg"

    # Salva temporariamente como JPEG
    original.save(
        temp_path,
        'JPEG',
        quality=quality
    )

    # Reabre imagem comprimida
    compressed = Image.open(
        temp_path
    )

    # Calcula diferença
    diff = ImageChops.difference(
        original,
        compressed
    )

    # Amplifica diferença
    extrema = diff.getextrema()

    max_diff = max(
        [ex[1] for ex in extrema]
    )

    scale = (
        255.0 / max_diff
        if max_diff != 0
        else 1
    )

    diff = ImageEnhance.Brightness(
        diff
    ).enhance(scale)

    # Converte para OpenCV
    ela_image = cv2.cvtColor(
        np.array(diff),
        cv2.COLOR_RGB2BGR
    )

    # Remove temporário
    os.remove(temp_path)

    return ela_image


# ============================================================
# CONFIGURAÇÃO DAS PASTAS
# ============================================================
output_root = "ela_results"
def run_base_imgs():

    folders = {
        "real": os.path.join(
            output_root,
            "real"
        ),

        "ia": os.path.join(
            output_root,
            "ia"
        ),

        "whatsapp": os.path.join(
            output_root,
            "whatsapp"
        ),

        "ia_whatsapp": os.path.join(
            output_root,
            "ia_whatsapp"
        )
    }


    # Cria todas as pastas
    for folder in folders.values():

        os.makedirs(
            folder,
            exist_ok=True
        )


    # ============================================================
    # PROCESSAR IMAGENS
    # ============================================================

    images = {

        "gustavo_real.jpg":
            (
                "real",
                "gustavo_real.jpg"
            ),

        "gustavo_ia_gemini.jpg":
            (
                "ia",
                "gustavo_ia_gemini.jpg"
            ),

        "gustavo_zap.jpg":
            (
                "whatsapp",
                "gustavo_zap.jpg"
            ),

        "gustavo_ia_gemini_zap.jpg":
            (
                "ia_whatsapp",
                "gustavo_ia_gemini_zap.jpg"
            )
    }


    for image_name, (
        category,
        image_path
    ) in images.items():

        print(
            f"\nProcessando: {image_name}"
        )

        # Calcula ELA
        ela = compute_ela(
            image_path
        )

        # Rotação específica da imagem real
        if category == "real":

            ela = cv2.rotate(
                ela,
                cv2.ROTATE_90_CLOCKWISE
            )

        # Nome do arquivo final
        original_name = os.path.splitext(
            image_name
        )[0]

        output_name = (
            original_name +
            "_ela.jpg"
        )

        output_path = os.path.join(
            folders[category],
            output_name
        )

        # Salva ELA
        cv2.imwrite(
            output_path,
            ela
        )

        # Analisa
        color_analysis = (
            count_gray_and_color_pixels(
                ela
            )
        )

        print(
            f"Gray ratio: "
            f"{color_analysis['gray_ratio']:.4f}"
        )

        print(
            f"Color ratio: "
            f"{color_analysis['color_ratio']:.4f}"
        )

        print(
            f"Black ratio: "
            f"{color_analysis['black_ratio']:.4f}"
        )

        print(
            f"gray_count: "
            f"{color_analysis['gray_count']}"
        )

        print(
            f"color_count: "
            f"{color_analysis['color_count']}"
        )

        print(
            f"black_count: "
            f"{color_analysis['black_count']}"
        )

        print(
            f"Salvo em: {output_path}"
        )


    print(
        "\nTodas as imagens ELA foram salvas!"
    )
def run_by_folder(_input_folder: str, _output_folder: str):
    qtd_processed = 0
    qtd_skipped = 0
    qtd_failed = 0
    for filename in os.listdir(_input_folder):
        image_path = os.path.join(_input_folder, filename)

        if not os.path.isfile(image_path):
            run_by_folder(image_path, os.path.join(_output_folder, filename))

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        # print(f"\nProcessando: {filename}")
        os.makedirs(
            _output_folder,
            exist_ok=True
        )
        original_name = os.path.splitext(filename)[0]
        output_name = original_name + "_ela.png"
        output_path = os.path.join(_output_folder, output_name)

        if os.path.exists(output_path):
            qtd_skipped += 1
            if qtd_skipped % 100 == 0:
                print(f"images skipped: {qtd_skipped}")
            continue
        try:
            ela = compute_ela(image_path)
            cv2.imwrite(output_path, ela)
        except Exception as e:
            qtd_failed += 1
            print(e)
            print(output_path)

        # color_analysis = count_gray_and_color_pixels(ela)
        qtd_processed+=1
        if qtd_processed % 500 == 0:
            print(f"images processed: {qtd_processed}")
    print(f"images processed: {qtd_processed}")
    print(f"images skipped: {qtd_skipped}")
    print(f"images failed: {qtd_failed}")

        # print(f"Gray ratio: {color_analysis['gray_ratio']:.4f}")
        # print(f"Color ratio: {color_analysis['color_ratio']:.4f}")
        # print(f"Black ratio: {color_analysis['black_ratio']:.4f}")
        # print(f"Salvo em: {output_path}")

if __name__ == "__main__":
    input_folder = "original"
    output_folder = "ela_results"
    run_by_folder(input_folder, output_folder)