import cv2
import matplotlib.pyplot as plt
import numpy as np
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
    # Citește imagine color
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # pentru PIL normal

    # Histograma grayscale
    fig1, ax1 = plt.subplots()
    ax1.hist(gray.ravel(), 256, [0, 256])
    ax1.set_title("Histograma imaginii in tonuri de gri")
    hist_gray_pil = fig_to_pil(fig1)
    plt.close(fig1)

    # Histograma color
    color = ('b', 'g', 'r')
    fig2, ax2 = plt.subplots()
    for i, c in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax2.plot(histr, color=c)
        ax2.set_xlim([0, 256])
    ax2.set_title("Histograma imaginii color")
    hist_color_pil = fig_to_pil(fig2)
    plt.close(fig2)

    results = [
        ("Original", cv2_to_pil(img, is_bgr=True)),
        ("Grayscale", Image.fromarray(gray)),
        ("Binary Image", Image.fromarray(bw)),
        ("HSV", cv2_to_pil(hsv_bgr, is_bgr=True)),
        ("Histogram Gray", hist_gray_pil),
        ("Histogram Color", hist_color_pil),
    ]
    return results
