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

def global_thresholding(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist=hist[:,0]
    total = image.size
    sumB,wB,maximum,sum1=0,0,0,np.dot(np.arange(256),hist)
    level=0
    for i in range(256):
        wB += hist[i]
        if wB==0:
            continue
        wF = total - wB
        if wF==0:
            break
        sumB+= i*hist[i]
        mB= sumB/wB
        mF= (sum1 - sumB)/wF
        between= wB*wF*(mB-mF)**2
        if between>maximum:
            level=i
            maximum=between
    _, binary_img = cv2.threshold(image, level, 255, cv2.THRESH_BINARY)
    return level, binary_img

def histogram_equalization(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist=hist.ravel()/image.size
    cdf= hist.cumsum()
    cdf_norm= cdf*255/cdf[-1]
    eq_img = np.interp(image.flatten(), np.arange(256), cdf_norm)
    eq_img= eq_img.reshape(image.shape).astype(np.uint8)
    return eq_img

def plot_images_and_histograms(original, processed, title_processed):
    fig, axes = plt.subplots(2,2, figsize=(12,6))

    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title("Imaginea originală")
    axes[0,0].axis('off')

    axes[0,1].hist(original.ravel(),256,[0,256])
    axes[0,1].set_title("Hist original")

    axes[1,0].imshow(processed, cmap='gray')
    axes[1,0].set_title(title_processed)
    axes[1,0].axis('off')

    axes[1,1].hist(processed.ravel(),256,[0,256])
    axes[1,1].set_title(f"Hist {title_processed}")

    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)
    return fig_pil

def process_image(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    # Binarizare
    level, binary_img = global_thresholding(img)
    fig_bin = plot_images_and_histograms(img, binary_img, f"Binarizată (Otsu={level})")

    # Egalizare
    eq_img = histogram_equalization(img)
    fig_eq = plot_images_and_histograms(img, eq_img, "Egalizată")

    results = [
        ("Original", Image.fromarray(img)),
        (f"Binarizată (Otsu={level})", Image.fromarray(binary_img)),
        ("Plot Binarizare", fig_bin),
        ("Egalizată", Image.fromarray(eq_img)),
        ("Plot Egalizare", fig_eq),
    ]
    return results
