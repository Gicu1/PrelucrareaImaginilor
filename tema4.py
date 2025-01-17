import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def cv2_to_pil(cv2_img, is_bgr=False):
    return Image.fromarray(cv2_img)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def morphological_dilation_fast(img):
    dil = np.zeros_like(img)
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            tmp = np.roll(np.roll(img, dx, axis=0), dy, axis=1)
            dil = np.logical_or(dil, tmp)
    return dil.astype(np.uint8)

def region_fill(bin_img, seed_coords=None):
    A_c = 1-bin_img
    X_k = np.zeros_like(bin_img, dtype=np.uint8)
    if seed_coords is None:
        seed_coords=[(0,0)]

    while True:
        for (sx,sy) in seed_coords:
            if 0<=sx<X_k.shape[0] and 0<=sy<X_k.shape[1]:
                X_k[sx,sy]=1
        X_dil = morphological_dilation_fast(X_k)
        X_kplus1 = np.logical_and(X_dil, A_c).astype(np.uint8)
        if np.array_equal(X_kplus1, X_k):
            break
        X_k = X_kplus1
    filled = np.logical_or(bin_img, 1-X_kplus1).astype(np.uint8)
    return filled

def process_image(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    _, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    bin01 = (binary//255).astype(np.uint8)

    dil = morphological_dilation_fast(bin01)
    contour_01 = (dil - bin01).clip(0,1)
    filled_01 = region_fill(bin01, seed_coords=[(0,0)])

    # subplot
    fig, axs = plt.subplots(1,4, figsize=(18,5))
    axs[0].imshow(img, cmap='gray'); axs[0].set_title("Imagine inițială"); axs[0].axis('off')
    axs[1].imshow(bin01, cmap='gray'); axs[1].set_title("Imagine binarizată"); axs[1].axis('off')
    axs[2].imshow(contour_01, cmap='gray'); axs[2].set_title("Contur"); axs[2].axis('off')
    axs[3].imshow(filled_01, cmap='gray'); axs[3].set_title("Filled"); axs[3].axis('off')
    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original Gray", Image.fromarray(img)),
        ("Binary 0/1", Image.fromarray((bin01*255).astype(np.uint8))),
        ("Contour", Image.fromarray((contour_01*255).astype(np.uint8))),
        ("Filled", Image.fromarray((filled_01*255).astype(np.uint8))),
        # ("Colaj (1x4)", fig_pil),
    ]
    return results
