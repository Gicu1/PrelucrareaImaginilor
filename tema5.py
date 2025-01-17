import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import io

def cv2_to_pil(cv2_img, is_bgr=False):
    if is_bgr:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def find(parent, x):
    if parent[x]!=x:
        parent[x]= find(parent, parent[x])
    return parent[x]

def two_pass_labeling(binary_image):
    rows,cols = binary_image.shape
    labels = np.zeros((rows,cols),dtype=int)
    label=1
    parent={}
    neighbors=[(-1,-1),(-1,0),(-1,1),(0,-1)]
    for i in range(rows):
        for j in range(cols):
            if binary_image[i,j]==0:
                neighbor_lbl=[]
                for dx,dy in neighbors:
                    nx,ny=i+dx,j+dy
                    if 0<=nx<rows and 0<=ny<cols and labels[nx,ny]>0:
                        neighbor_lbl.append(labels[nx,ny])
                if not neighbor_lbl:
                    labels[i,j]=label
                    parent[label]=label
                    label+=1
                else:
                    mlabel=min(neighbor_lbl)
                    labels[i,j]=mlabel
                    for lbl in neighbor_lbl:
                        r1=find(parent,lbl)
                        r2=find(parent,mlabel)
                        if r1!=r2:
                            parent[r2]=r1
    for i in range(rows):
        for j in range(cols):
            if labels[i,j]>0:
                labels[i,j]= find(parent,labels[i,j])
    unique_lbl=np.unique(labels)
    unique_lbl=unique_lbl[unique_lbl!=0]
    new_label=1
    mapping={}
    for lbl in unique_lbl:
        mapping[lbl]=new_label
        new_label+=1
    for i in range(rows):
        for j in range(cols):
            if labels[i,j] in mapping:
                labels[i,j]=mapping[labels[i,j]]
    return labels,new_label-1

def label_to_color(labels, num_labels):
    colored = np.ones((labels.shape[0],labels.shape[1],3),dtype=np.float32)
    cmap = plt.cm.get_cmap('nipy_spectral', num_labels+1)
    for lbl in range(1,num_labels+1):
        colored[labels==lbl]= cmap(lbl)[:3]
    colored=(colored*255).astype(np.uint8)
    return colored

def moore_neighbor_tracing(labels, label_num):
    rows,cols = labels.shape
    contour=[]
    found=False
    for i in range(rows):
        for j in range(cols):
            if labels[i,j]== label_num:
                start=(i,j)
                found=True
                break
        if found: break
    if not found:
        return contour

    directions=[(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    current=start
    contour.append(current)
    b_dir=6
    while True:
        dir_idx=(b_dir+1)%8
        found_next=False
        for _ in range(8):
            dx,dy=directions[dir_idx]
            nx,ny = current[0]+dx,current[1]+dy
            if 0<=nx<rows and 0<=ny<cols and labels[nx,ny]==label_num:
                contour.append((nx,ny))
                current=(nx,ny)
                b_dir=(dir_idx+4)%8
                found_next=True
                break
            else:
                dir_idx=(dir_idx+1)%8
        if not found_next:
            break
        if current==start:
            break
    return contour

def extract_chain_code(contour):
    directions=[(-1,0),(-1,1),(0,1),(1,1),
                (1,0),(1,-1),(0,-1),(-1,-1)]
    chain_code=[]
    for i in range(1,len(contour)):
        prev=contour[i-1]
        curr=contour[i]
        dx= curr[0]-prev[0]
        dy= curr[1]-prev[1]
        try:
            code=directions.index((dx,dy))
            chain_code.append(code)
        except ValueError:
            chain_code.append(-1)
    return chain_code

def visualize_contours(image, labels, num_labels, contours):
    black_image = np.zeros((image.shape[0], image.shape[1], 3),dtype=np.uint8)
    for lbl_num in range(1, num_labels+1):
        ctr = contours.get(lbl_num, [])
        for (x,y) in ctr:
            black_image[x,y]=(255,255,255)
    return black_image

def process_image(input_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Nu s-a putut încărca imaginea: " + input_path)

    _, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    labels, num_labels = two_pass_labeling(binary)
    colored_labels= label_to_color(labels, num_labels)

    # Obținem contururile
    contours={}
    chain_codes={}
    for lbl in range(1, num_labels+1):
        contour = moore_neighbor_tracing(labels, lbl)
        contours[lbl]= contour
        c_code = extract_chain_code(contour)
        chain_codes[lbl]= c_code
        # Poți face print direct aici

    # Imagine cu contururi
    image_with_contours = visualize_contours(image, labels, num_labels, contours)
    image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)

    # Subplot
    fig, axs = plt.subplots(2,2, figsize=(20,10))
    axs[0,0].imshow(image, cmap='gray'); axs[0,0].set_title("Imagine Originală"); axs[0,0].axis('off')
    axs[0,1].imshow(binary, cmap='gray'); axs[0,1].set_title("Imagine Binarizată"); axs[0,1].axis('off')
    axs[1,0].imshow(colored_labels); axs[1,0].set_title(f"Obiecte Etichetate ({num_labels})"); axs[1,0].axis('off')
    axs[1,1].imshow(image_with_contours_rgb); axs[1,1].set_title("Contururi"); axs[1,1].axis('off')
    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    # (Opțional) afișăm în consolă chain codes
    for lbl in chain_codes:
        print(f"Chain Code pentru obiectul {lbl}: {chain_codes[lbl]}")

    results = [
        ("Original Gray", Image.fromarray(image)),
        ("Binary", Image.fromarray(binary)),
        (f"Labeled {num_labels} obj", cv2_to_pil(colored_labels, is_bgr=False)),
        ("Contours", cv2_to_pil(image_with_contours_rgb, is_bgr=False)),
        # ("Colaj Subplot", fig_pil),
    ]
    return results
