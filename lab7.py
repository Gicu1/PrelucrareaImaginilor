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
    # Citește color
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1.4)

    # Sobel
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F,0,1,ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    # NMS
    nms_img = np.zeros_like(magnitude, dtype=np.float32)
    angle = np.degrees(direction)
    angle[angle<0]+=180
    H,W = magnitude.shape
    for i in range(1,H-1):
        for j in range(1,W-1):
            a=angle[i,j]
            m=magnitude[i,j]
            if (0<=a<22.5) or (157.5<=a<180):
                n1 = magnitude[i,j-1]
                n2 = magnitude[i,j+1]
            elif (22.5<=a<67.5):
                n1 = magnitude[i-1,j-1]
                n2 = magnitude[i+1,j+1]
            elif (67.5<=a<112.5):
                n1 = magnitude[i-1,j]
                n2 = magnitude[i+1,j]
            else:
                n1 = magnitude[i-1,j+1]
                n2 = magnitude[i+1,j-1]
            if m>=n1 and m>=n2:
                nms_img[i,j]=m

    # Threshold adaptiv
    max_val = np.max(nms_img)
    high_thresh = 0.15*max_val
    low_thresh = 0.05*max_val

    bin_img = np.zeros_like(nms_img, dtype=np.uint8)
    strong_i, strong_j = np.where(nms_img>=high_thresh)
    weak_i, weak_j = np.where((nms_img<high_thresh)&(nms_img>=low_thresh))
    bin_img[strong_i, strong_j]=255
    bin_img[weak_i, weak_j]=128

    # Histereză
    final_img = bin_img.copy()
    stack=[(x,y) for x,y in zip(strong_i,strong_j)]
    neighbors=[(-1,-1),(-1,0),(-1,1),
               (0,-1),       (0,1),
               (1,-1),(1,0),(1,1)]
    while stack:
        x,y=stack.pop()
        for dx,dy in neighbors:
            nx,ny=x+dx,y+dy
            if 0<=nx<H and 0<=ny<W:
                if final_img[nx,ny]==128:
                    final_img[nx,ny]=255
                    stack.append((nx,ny))
    final_img[final_img!=255]=0

    # Subplot
    fig, axes = plt.subplots(1,3, figsize=(12,6))
    axes[0].imshow(img_gray, cmap='gray'); axes[0].set_title("Imagine Gray"); axes[0].axis('off')
    axes[1].imshow(bin_img, cmap='gray'); axes[1].set_title("NMS + Binarizare"); axes[1].axis('off')
    axes[2].imshow(final_img, cmap='gray'); axes[2].set_title("Rezultat final"); axes[2].axis('off')
    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original (BGR)", cv2_to_pil(img_bgr, is_bgr=True)),
        ("Grayscale", Image.fromarray(img_gray)),
        ("Bin Img (NMS+Adaptive)", Image.fromarray(bin_img)),
        ("Final Edges (Hysteresis)", Image.fromarray(final_img)),
        # ("Colaj (3 imagini)", fig_pil),
    ]
    return results
