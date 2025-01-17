import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import io

def cv2_to_pil(cv2_img, is_bgr=False):
    return Image.fromarray(cv2_img)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def find(parent, x):
    if parent[x]!=x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def bfs_labeling(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=int)
    label=1
    directions = [(-1,-1),(-1,0),(-1,1),
                  (0,-1),        (0,1),
                  (1,-1),(1,0),(1,1)]
    for i in range(rows):
        for j in range(cols):
            if binary_image[i,j]==0 and labels[i,j]==0:
                queue=deque()
                queue.append((i,j))
                labels[i,j]=label
                while queue:
                    x,y=queue.popleft()
                    for dx,dy in directions:
                        nx,ny = x+dx,y+dy
                        if 0<=nx<rows and 0<=ny<cols:
                            if binary_image[nx,ny]==0 and labels[nx,ny]==0:
                                labels[nx,ny]=label
                                queue.append((nx,ny))
                label+=1
    return labels,label-1

def two_pass_labeling(binary_image):
    rows,cols = binary_image.shape
    labels = np.zeros((rows,cols),dtype=int)
    label=1
    parent={}
    neighbors=[(-1,-1),(-1,0),(-1,1),
               (0,-1)]
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
                    mlabel = min(neighbor_lbl)
                    labels[i,j]=mlabel
                    for lbl in neighbor_lbl:
                        r1=find(parent,lbl)
                        r2=find(parent,mlabel)
                        if r1!=r2:
                            parent[r2]=r1
    for i in range(rows):
        for j in range(cols):
            if labels[i,j]>0:
                labels[i,j]=find(parent,labels[i,j])

    unique_lbl = np.unique(labels)
    unique_lbl = unique_lbl[unique_lbl!=0]
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
        colored[labels==lbl] = cmap(lbl)[:3]
    colored = (colored*255).astype(np.uint8)
    return colored

def process_image(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagine inexistentă: " + input_path)

    _, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    labels_bfs, nb_bfs = bfs_labeling(binary)
    labels_tp, nb_tp = two_pass_labeling(binary)

    bfs_col = label_to_color(labels_bfs, nb_bfs)
    tp_col = label_to_color(labels_tp, nb_tp)

    # Subplot
    fig, axes = plt.subplots(1,4, figsize=(24,6))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title("Imagine Originală"); axes[0].axis('off')
    axes[1].imshow(binary, cmap='gray'); axes[1].set_title("Imagine Binarizată"); axes[1].axis('off')
    axes[2].imshow(bfs_col); axes[2].set_title(f"BFS - {nb_bfs} obiecte"); axes[2].axis('off')
    axes[3].imshow(tp_col); axes[3].set_title(f"Two-Pass - {nb_tp} obiecte"); axes[3].axis('off')
    plt.tight_layout()
    fig_pil = fig_to_pil(fig)
    plt.close(fig)

    results = [
        ("Original (Grayscale)", Image.fromarray(img)),
        ("Binary", Image.fromarray(binary)),
        (f"BFS Labeling ({nb_bfs} obiecte)", cv2_to_pil(bfs_col, is_bgr=False)),
        (f"Two-Pass Labeling ({nb_tp} obiecte)", cv2_to_pil(tp_col, is_bgr=False)),
        # ("Colaj BFS/TwoPass", fig_pil),
    ]
    return results
