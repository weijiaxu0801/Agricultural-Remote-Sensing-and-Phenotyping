# Agricultural Remote Sensing & High-Throughput Phenotyping
This repository contains my Python pipelines for processing UAV RGB and airborne hyperspectral imagery to predict crop physiological traits. 

## 1. 0-high throughout leaf analysis random forest rgb
- **Task:** Estimate Leaf Area Index (LAI) and predict SPAD (chlorophyll proxy) using RGB imagery and random forest model.
- **Method:** 
  - Extracted vegetation indices (ExG, NGRDI, GLI).
  - Performed image segmentation (Otsu thresholding + morphological operations) to calculate canopy coverage (88%) and estimate LAI (4.23).
  - Built a Random Forest regression model using 5 indices (ExG, NGRDI, GLI, VARI, TGI) to predict SPAD across 11 field treatments. Achieved an **\(R^2\) of 0.455**.
- **Key insight:** Feature importance analysis identified VARI (Visible Atmospherically Resistant Index) as the most critical predictor.

## 2. 0-multiple time hyperspectral analysis new
- **Task:** Predict SPAD from airborne hyperspectral imagery across multiple dates.
- **Method:**
  - Extracted ground truth points from ESRI Shapefiles and matched them with hyperspectral pixels using GeoPandas and Rasterio.
  - Applied Savitzky-Golay first-derivative preprocessing to enhance spectral features in the Red-Edge and NIR regions.
  - Merged multi-date datasets and trained a Random Forest model via Leave-One-Out cross-validation. Achieved an **\(R^2\) of 0.153** (N = 335).
- **Key insight:** Analyzed model limitations caused by 30m spatial resolution (mixed pixels) and temporal mismatch. Feature importance highlighted critical bands in the Blue-Green and Red-Edge regions.

## 🛠 Tech Stack
- **Languages:** Python
- **Libraries:** OpenCV, Scikit-learn, Rasterio, GeoPandas, NumPy, Pandas, Matplotlib, SciPy
