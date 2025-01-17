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
        raise ValueError("Imagine inexistentă: " + input_path)

    # Filtre
    kernel_mean_3 = np.ones((3,3), np.float32)/9
    filtered_mean_3 = cv2.filter2D(img, -1, kernel_mean_3)

    kernel_mean_5 = np.ones((5,5), np.float32)/25
    filtered_mean_5 = cv2.filter2D(img, -1, kernel_mean_5)

    kernel_mean_7 = np.ones((7,7), np.float32)/49
    filtered_mean_7 = cv2.filter2D(img, -1, kernel_mean_7)

    kernel_mean_9 = np.ones((9,9), np.float32)/81
    filtered_mean_9 = cv2.filter2D(img, -1, kernel_mean_9)

    kernel_gaussian_3 = np.array([[1,2,1],
                                  [2,4,2],
                                  [1,2,1]], dtype=np.float32)/16
    filtered_gaussian = cv2.filter2D(img, -1, kernel_gaussian_3)

    kernel_laplace_1 = np.array([[0,-1,0],
                                 [-1,4,-1],
                                 [0,-1,0]], dtype=np.float32)
    kernel_laplace_2 = np.array([[-1,-1,-1],
                                 [-1,8,-1],
                                 [-1,-1,-1]], dtype=np.float32)

    filtered_laplace_1 = cv2.filter2D(img, -1, kernel_laplace_1)
    filtered_laplace_2 = cv2.filter2D(img, -1, kernel_laplace_2)

    # Subplot colaj
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title("Original"); axes[0,0].axis('off')
    axes[0,1].imshow(filtered_mean_3, cmap='gray'); axes[0,1].set_title("Mean 3x3"); axes[0,1].axis('off')
    axes[0,2].imshow(filtered_mean_5, cmap='gray'); axes[0,2].set_title("Mean 5x5"); axes[0,2].axis('off')
    axes[0,3].imshow(filtered_mean_7, cmap='gray'); axes[0,3].set_title("Mean 7x7"); axes[0,3].axis('off')
    axes[1,0].imshow(filtered_mean_9, cmap='gray'); axes[1,0].set_title("Mean 9x9"); axes[1,0].axis('off')
    axes[1,1].imshow(filtered_gaussian, cmap='gray'); axes[1,1].set_title("Gaussian 3x3"); axes[1,1].axis('off')
    axes[1,2].imshow(filtered_laplace_1, cmap='gray'); axes[1,2].set_title("Laplace 1"); axes[1,2].axis('off')
    axes[1,3].imshow(filtered_laplace_2, cmap='gray'); axes[1,3].set_title("Laplace 2"); axes[1,3].axis('off')

    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original", Image.fromarray(img)),
        ("Mean 3x3", Image.fromarray(filtered_mean_3)),
        ("Mean 5x5", Image.fromarray(filtered_mean_5)),
        ("Mean 7x7", Image.fromarray(filtered_mean_7)),
        ("Mean 9x9", Image.fromarray(filtered_mean_9)),
        ("Gaussian 3x3", Image.fromarray(filtered_gaussian)),
        ("Laplace Filtered 1", Image.fromarray(filtered_laplace_1)),
        ("Laplace Filtered 2", Image.fromarray(filtered_laplace_2)),
        # ("Colaj (2x4)", fig_pil),
    ]
    return results
