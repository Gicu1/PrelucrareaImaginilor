import cv2
import os
from PIL import Image
import numpy as np
import io

from PIL import Image
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
    # Citește imaginea
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Imagine inexistentă: " + input_path)

    # Directorul și salvarea imaginii
    directory = os.path.dirname(input_path)
    filename = 'SavedFlower.jpg'
    saved_path = os.path.join(directory, filename)
    cv2.imwrite(saved_path, img)

    # Returnăm două imagini:
    # 1) Original (color)
    # 2) "SavedFlower.jpg" reîncărcată (ca dovadă că e salvată)
    saved_img = cv2.imread(saved_path)  # tot BGR
    results = [
        ("Original Image", cv2_to_pil(img, is_bgr=True)),
        ("Saved Flower", cv2_to_pil(saved_img, is_bgr=True)),
    ]
    return results
