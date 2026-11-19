# This function counts the vehicles of each class when I run the inference
 
from src.config import SMALL, LARGE
 
 
def count_vehicles(classes):
    
    small = int((classes == SMALL).sum())
    large = int((classes == LARGE).sum())
    return {"small": small, "large": large, "total": small + large}