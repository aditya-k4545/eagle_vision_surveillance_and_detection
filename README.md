# Aerial Traffic Monitoring and Density Analysis with YOLO26 OBB

Detects and counts vehicles in aerial images using oriented bounding boxes, and maps traffic density on a grid. A YOLO26-OBB model is trained on the DLR EAGLE dataset (converted to YOLO OBB format and tiled so small vehicles stay visible), with a Streamlit demo.

**Live demo:** https://eaglevisionsurveillanceanddetection.streamlit.app/ 

## Results

Validation on 1024 x 1024 tiles of the EAGLE validation split :

| Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-------|----------|
| 0.762     | 0.743  | 0.762 | 0.580    |


## Features

- Rotated boxes that follow each vehicle's direction
- Two classes: small (cars, vans) and large (trucks, buses, trailers)
- Traffic density heatmap over an adjustable grid
- Streamlit app: upload an image, set confidence and grid size, download the result

## Project structure

```
app.py          Streamlit interface
src/            detection, counting, config and density logic
models/         trained weights (best.pt)
results/        training curves 
```

## Dataset

The dataset is **not included** in this repository. EAGLE is published by the German Aerospace Center (DLR) and must be downloaded from DLR under its own terms: [dataset page](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/eagle).

EAGLE uses its own XML annotation format, with extra information such as vehicle orientation and visibility flags. After downloading, the labels must be converted to YOLO OBB format (and the images tiled) before training.

Paper: *EAGLE: Large-scale Vehicle Detection Dataset in Real-World Scenarios using Aerial Imagery*, ICPR 2020 (arXiv:2007.06124).



