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

def normalizeaza_histograma(hist):
    return hist / np.sum(hist)

def gaseste_maxime(fdp, WH=5, TH=0.0003):
    maxime = []
    for k in range(WH, 256-WH):
        fereastra = fdp[k-WH:k+WH+1]
        v = np.mean(fereastra)
        if fdp[k]>v+TH and fdp[k]>=np.max(fereastra):
            maxime.append(k)
    maxime=[0]+maxime+[255]
    return maxime

def calculeaza_praguri(maxime):
    praguri=[]
    for i in range(len(maxime)-1):
        prag=(maxime[i]+maxime[i+1])//2
        praguri.append(prag)
    return praguri

def gaseste_cel_mai_apropiat_maxim(pixel, maxime):
    return min(maxime, key=lambda x: abs(x-pixel))

def floyd_steinberg_dithering(imagine, maxime):
    h,w=imagine.shape
    out = imagine.astype(float)
    for y in range(h-1):
        for x in range(1, w-1):
            old_val = out[y,x]
            new_val = gaseste_cel_mai_apropiat_maxim(old_val, maxime)
            out[y,x] = new_val
            eroare = old_val - new_val
            out[y, x+1]+= eroare*(7/16)
            out[y+1, x-1]+= eroare*(3/16)
            out[y+1, x]+= eroare*(5/16)
            out[y+1, x+1]+= eroare*(1/16)
    return np.clip(out,0,255).astype(np.uint8)

def process_image(input_path):
    imagine = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if imagine is None:
        raise ValueError("Nu s-a putut citi imaginea: " + input_path)

    hist = cv2.calcHist([imagine],[0],None,[256],[0,256]).ravel()
    fdp = normalizeaza_histograma(hist)
    maxime = gaseste_maxime(fdp)
    praguri = calculeaza_praguri(maxime)

    # Cuantizare simplă
    imagine_cuantizata = np.zeros_like(imagine)
    for i in range(imagine.shape[0]):
        for j in range(imagine.shape[1]):
            imagine_cuantizata[i,j] = gaseste_cel_mai_apropiat_maxim(imagine[i,j], maxime)

    # Dithering pe imaginea cuantizată
    imagine_dithering = floyd_steinberg_dithering(imagine_cuantizata.copy(), maxime)

    # Dithering direct pe original
    imagine_dithering_original = floyd_steinberg_dithering(imagine.copy(), maxime)

    # Creăm colaj identic cu codul original
    fig, axs = plt.subplots(2,3, figsize=(12,8))

    axs[0,0].imshow(imagine, cmap='gray')
    axs[0,0].set_title('Imagine originala')

    axs[0,1].imshow(imagine_cuantizata, cmap='gray')
    axs[0,1].set_title('Imagine cuantizata')

    axs[0,2].imshow(imagine_dithering, cmap='gray')
    axs[0,2].set_title('Imagine dupa dithering')

    axs[1,0].plot(hist)
    axs[1,0].set_title('Histograma originala')

    axs[1,1].plot(fdp)
    # marcam maxime
    axs[1,1].plot(maxime, fdp[maxime], 'ro', label='Maxime')
    for prag in praguri:
        axs[1,1].axvline(x=prag, color='g', linestyle='--', alpha=0.5)
    axs[1,1].set_title('FDP cu maxime si praguri')
    axs[1,1].legend()

    # subplot(1,2) cu text
    axs[1,2].axis('off')
    textstr = f"Maxime:\n{maxime}\n\nPraguri:\n{praguri}"
    axs[1,2].text(0.5,0.5, textstr, fontsize=12, va='center', ha='center')
    axs[1,2].set_title('Maxime si Praguri')

    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original Gray", Image.fromarray(imagine)),
        ("Cuantizata", Image.fromarray(imagine_cuantizata)),
        ("Dithering on Cuantizata", Image.fromarray(imagine_dithering)),
        ("Dithering direct Original", Image.fromarray(imagine_dithering_original)),
        # ("Colaj Subplot", fig_pil),
    ]
    return results
