# This file contains the functions [shrink,detect] which shrinks the image if it is too large before giving to the model and then gives it to detect function for gettig predictions

import numpy as np
from PIL import Image
import cv2
 
 
def shrink(image, max_side=2048):
    
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
 
 
def detect(model, image, conf=0.25, img_size=1024, max_det=2000):
   
    result = model.predict(
        image, imgsz=img_size, conf=conf, max_det=max_det, verbose=False
    )[0]
 
    annotated = result.plot(labels=False, conf=False)   
     
    if result.obb is None or len(result.obb) == 0:
        return annotated, np.zeros((0, 2)), np.zeros(0, dtype=int)
 
    polygons = result.obb.xyxyxyxy.cpu().numpy()          
    classes = result.obb.cls.cpu().numpy().astype(int)    
    centers = polygons.mean(axis=1)                       
    
    return annotated, centers, classes