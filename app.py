import io
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import ultralytics
from ultralytics import YOLO
 
from src.counting import count_vehicles
from src.density import density_grids, density_figure
from src.detection import detect, shrink

 
ROOT = Path(__file__).parent
MODEL_PATH =ROOT / "models" / "best.pt"
SAMPLE_PATH = ROOT / "sample_inference_img.png"
 
IMG_SIZE = 1024    
MAX_SIDE = 2048    
MAX_DET = 2000     
 
st.set_page_config(page_title="Aditya's Aerial Surveillance & Traffic Detection", layout="wide")
 
 
@st.cache_resource
def load_model():
    
    return YOLO(str(MODEL_PATH))
 
 
def grid_table(grid):

    return pd.DataFrame(
        grid,
        index=[f"Row {i + 1}" for i in range(grid.shape[0])],
        columns=[f"Col {j + 1}" for j in range(grid.shape[1])],
    )
 

 
st.title("Aditya's Aerial Surveillance & Traffic Detection")
st.divider()
st.caption(
    "YOLO26 model trained on the modified DLR's EAGLE dataset with tiling "
    
)
 
with st.sidebar:
    st.header("Choose your desired Configuration")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    rows = st.slider("Grid rows", 2, 10, 4)
    cols = st.slider("Grid columns", 2, 10, 4)
 
uploaded = st.file_uploader("Upload aerial image", type=["jpg", "jpeg", "png"])
 
image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif SAMPLE_PATH.exists() and st.checkbox("Use the sample image"):
    image = Image.open(SAMPLE_PATH).convert("RGB")
 
if image is not None:
    image = shrink(image, MAX_SIDE)
 
    st.subheader("Original image")
    st.image(image)
 
    if st.button("Run detection", type="primary"):
        model = load_model()
 
        with st.spinner("Detecting vehicles (CPU inference can take a while)..."):
            annotated, centers, classes = detect(
                model, image, conf, IMG_SIZE, MAX_DET
            )
 
        counts = count_vehicles(classes)
        width, height = image.size
        grids = density_grids(centers, classes, width, height, rows, cols)
 
        st.subheader("Vehicle count")
        m1, m2, m3 = st.columns(3)
        m1.metric("Small vehicles", counts["small"])
        m2.metric("Large vehicles", counts["large"])
        m3.metric("Total vehicles", counts["total"])
 
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fig = density_figure(rgb, grids["total"])
 
        st.subheader("Detections with traffic density grid")
        st.caption("Each cell shows its vehicle count. Redder cells are busier.")
        st.pyplot(fig)
 
        if grids["total"].max() > 0:
            busiest = np.unravel_index(grids["total"].argmax(), grids["total"].shape)
            st.info(
                f"Busiest cell: row {busiest[0] + 1}, column {busiest[1] + 1} "
                f"with {grids['total'].max()} vehicles."
            )
 
        st.subheader("Vehicle density tables")
        t1, t2, t3 = st.tabs(["All_Vehicles", "Small_Vehicles", "Large_Vehicles"])
        t1.dataframe(grid_table(grids["total"]))
        t2.dataframe(grid_table(grids["small"]))
        t3.dataframe(grid_table(grids["large"]))