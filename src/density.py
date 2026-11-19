# This code assigns the counted vehicles to grids to calculate the the Traffic Density and also monts the heatmap on the annotated image
 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
 
from src.config import SMALL, LARGE
 
 
def density_grids(centers, classes, width, height, rows, cols):

    grids = {}
    for k in ("total","small","large"):
        grids[k] = np.zeros((rows,cols),dtype=int)
    
    if len(centers) == 0:
        return grids
 
    
    c = np.clip((centers[:, 0] / width * cols).astype(int), 0, cols - 1)
    r = np.clip((centers[:, 1] / height * rows).astype(int), 0, rows - 1)
 
    np.add.at(grids["total"], (r, c), 1)
    np.add.at(grids["small"], (r[classes == SMALL], c[classes == SMALL]), 1)
    np.add.at(grids["large"], (r[classes == LARGE], c[classes == LARGE]), 1)
    return grids
 
 
def density_figure(image_rgb, counts):
    
    rows, cols = counts.shape
    h, w = image_rgb.shape[:2]
 
    fig = plt.figure(figsize=(10, 10 * h / w))
    ax = fig.subplots()
    ax.imshow(image_rgb, extent=(0, cols, rows, 0), aspect="auto")
    sns.heatmap(counts, annot=True, fmt="d", cmap="YlOrRd", alpha=0.60,
                linewidths=1.5, linecolor="white", cbar=False,
                xticklabels=False, yticklabels=False, ax=ax)

    return fig
 