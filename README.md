# Agricultural Remote Sensing & High-Throughput Phenotyping

Two Python pipelines for processing UAV RGB and airborne hyperspectral imagery to predict crop physiological traits.

**Data sources.** The RGB imagery and SPAD measurements come from my own graduate research project.
The hyperspectral imagery (4 dates) comes from the public **Neusling 2012** dataset archived by GFZ / WDC-terra (DOI: 10.5880/enmap.2016.007) — not my own fieldwork; my contribution was the metadata handling, point–pixel matching, spectral preprocessing, modelling and validation.

**Before running:** the image data is not included in this repository. Download it from the DOI above, then replace the eight placeholder paths in `datasets` (`path/to/your/...`) with your own local paths.


## 1. RGB high-throughput leaf analysis → LAI & SPAD(0-high throughput leaf analysis random forest rgb.py)

**Task**: Estimate Leaf Area Index (LAI) and predict SPAD (chlorophyll proxy) using RGB imagery and a random forest model.
**Method**:
  Extracted vegetation indices (ExG, NGRDI, GLI).
  Performed image segmentation (Otsu thresholding + morphological operations) to calculate canopy coverage (88%) and estimate LAI (4.23).
  Built a Random Forest regression model using 5 indices (ExG, NGRDI, GLI, VARI, TGI) to predict SPAD across 11 field treatments. Achieved an **R² of 0.455**.
  **Key insight**: Feature importance analysis identified VARI (Visible Atmospherically Resistant Index) as the most critical predictor.

## 2. Multi-date hyperspectral SPAD inversion(0-multiple time hyperspectral analysis new.py)

 **Task**: Predict SPAD from airborne hyperspectral imagery across multiple dates.
 **Method**:
   Extracted ground truth points from ESRI Shapefiles and matched them with hyperspectral pixels using GeoPandas and Rasterio.
   Applied Savitzky–Golay first-derivative preprocessing to enhance spectral features in the red-edge and NIR regions.
   Merged multi-date datasets and trained a Random Forest model via Leave-One-Out cross-validation. Achieved an **R² of 0.153** (N = 335).
 **Sensor note**: the imagery is **airborne AVIS-3 at 4 m ground sampling distance** (197 bands, 477–1704 nm), read from the ENVI header — not EnMAP satellite data, and not 30 m.
 **Key insight**: The modest accuracy stems mainly from the mismatch between a 4 m pixel and a ~1 cm² leaf-level SPAD measurement, plus multi-date radiometric differences and the absence of soil-background masking. Feature importance highlighted bands in the blue–green and red-edge regions.

## 🛠 Tech Stack

 **Languages**: Python
 **Libraries**: OpenCV, Scikit-learn, Rasterio, GeoPandas, NumPy, Pandas, Matplotlib, SciPy
