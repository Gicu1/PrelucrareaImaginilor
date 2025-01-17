import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import io

def cv2_to_pil(cv2_img, is_bgr=False):
    return Image.fromarray(cv2_img)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size//2), size//2, size)
    gauss = np.exp(-0.5*(ax/sigma)**2)
    kernel = np.outer(gauss, gauss)
    return kernel/np.sum(kernel)

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def process_image(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    # Zgomot
    noisy_img = add_gaussian_noise(img)

    # Filtru Gaussian
    g_kernel = gaussian_kernel(5, 1.0)
    start_time = time.time()
    gaussian_filtered = cv2.filter2D(noisy_img, -1, g_kernel)
    gauss_time = time.time() - start_time

    # Kernel bidimensional 3x3
    b_kernel = np.ones((3,3), dtype=np.float32)/9
    start_time = time.time()
    bidimensional_filtered = cv2.filter2D(noisy_img, -1, b_kernel)
    bidim_time = time.time() - start_time

    # Subplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(noisy_img, cmap='gray'); axes[0].set_title("Zgomot Gaussian"); axes[0].axis('off')
    axes[1].imshow(gaussian_filtered, cmap='gray')
    axes[1].set_title(f"Filtru Gaussian\nTimp: {gauss_time:.4f} s")
    axes[1].axis('off')
    axes[2].imshow(bidimensional_filtered, cmap='gray')
    axes[2].set_title(f"Filtru Bidimensional\nTimp: {bidim_time:.4f} s")
    axes[2].axis('off')

    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original Grayscale", Image.fromarray(img)),
        ("Noisy", Image.fromarray(noisy_img)),
        (f"Gaussian Filtered ({gauss_time:.4f}s)", Image.fromarray(gaussian_filtered)),
        (f"Bidimensional Filter ({bidim_time:.4f}s)", Image.fromarray(bidimensional_filtered)),
        # ("Colaj (3 imagini)", fig_pil),
    ]
    return results
