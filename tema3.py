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

def process_image(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows,cols = img.shape
    crow,ccol = rows//2, cols//2
    sigma=30

    # Gaussian Low Pass
    glp = np.zeros((rows,cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i-crow)**2+(j-ccol)**2)
            glp[i,j]= np.exp(-(dist**2)/(2*sigma**2))
    glp_dft = dft_shift*glp
    idft_glp = np.fft.ifftshift(glp_dft)
    img_glp = np.abs(np.fft.ifft2(idft_glp))

    # Gaussian High Pass
    ghp = 1- glp
    ghp_dft = dft_shift*ghp
    idft_ghp = np.fft.ifftshift(ghp_dft)
    img_ghp = np.abs(np.fft.ifft2(idft_ghp))

    # Ideal Low Pass
    D0=30
    ilp = np.zeros((rows,cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i-crow)**2+(j-ccol)**2)
            if dist<=D0:
                ilp[i,j]=1
    ilp_dft= dft_shift*ilp
    idft_ilp= np.fft.ifftshift(ilp_dft)
    img_ilp= np.abs(np.fft.ifft2(idft_ilp))

    # Ideal High Pass
    ihp=1-ilp
    ihp_dft= dft_shift*ihp
    idft_ihp= np.fft.ifftshift(ihp_dft)
    img_ihp= np.abs(np.fft.ifft2(idft_ihp))

    # subplot
    fig, axes = plt.subplots(2,3, figsize=(12,8))
    axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title("Original"); axes[0,0].axis('off')
    axes[0,1].imshow(img_glp, cmap='gray'); axes[0,1].set_title("Gaussian LP"); axes[0,1].axis('off')
    axes[0,2].imshow(img_ghp, cmap='gray'); axes[0,2].set_title("Gaussian HP"); axes[0,2].axis('off')
    axes[1,0].imshow(img_ilp, cmap='gray'); axes[1,0].set_title("Ideal LP"); axes[1,0].axis('off')
    axes[1,1].imshow(img_ihp, cmap='gray'); axes[1,1].set_title("Ideal HP"); axes[1,1].axis('off')
    # Ultimul subplot e liber
    axes[1,2].axis('off')

    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original", Image.fromarray(img.astype(np.uint8))),
        ("Gaussian Low Pass", Image.fromarray(img_glp.astype(np.uint8))),
        ("Gaussian High Pass", Image.fromarray(img_ghp.astype(np.uint8))),
        ("Ideal Low Pass", Image.fromarray(img_ilp.astype(np.uint8))),
        ("Ideal High Pass", Image.fromarray(img_ihp.astype(np.uint8))),
        # ("Colaj (2x3)", fig_pil),
    ]
    return results
