import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def cv2_to_pil(cv2_img, is_bgr=True):
    if is_bgr:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def process_image(input_path):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    negative_img = 255 - img
    contrast_img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
    gamma = 3
    invGamma = 1.0 / gamma
    table = np.array([( (i/255.0)**invGamma )*255 for i in range(256)], dtype=np.uint8)
    gamma_img = cv2.LUT(img, table)
    brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=60)

    # Convertim la RGB pentru a fi afișate corect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    negative_rgb = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)
    contrast_rgb = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB)
    gamma_rgb = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
    brightness_rgb = cv2.cvtColor(brightness_img, cv2.COLOR_BGR2RGB)

    # Subplot colaj
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(negative_rgb)
    axes[0, 1].set_title('Negative Image')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(contrast_rgb)
    axes[0, 2].set_title('Contrast Image')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(brightness_rgb)
    axes[1, 0].set_title('Brightness')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gamma_rgb)
    axes[1, 1].set_title('Gamma Correction')
    axes[1, 1].axis('off')

    # axes[1, 2] e nefolosit - îl ascundem
    axes[1, 2].axis('off')

    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original", Image.fromarray(img_rgb)),
        ("Negative", Image.fromarray(negative_rgb)),
        ("Contrast", Image.fromarray(contrast_rgb)),
        ("Gamma", Image.fromarray(gamma_rgb)),
        ("Brightness", Image.fromarray(brightness_rgb)),
        # ("Subplot (colaj)", fig_pil),
    ]
    return results
