# Agricultural Remote Sensing & High-Throughput Phenotyping

Two Python pipelines for processing UAV RGB and airborne hyperspectral imagery to predict crop physiological traits.

**Data sources.** The RGB imagery and SPAD measurements come from my own graduate research project.
The hyperspectral imagery (4 dates) comes from the public **Neusling 2012** dataset archived by GFZ / WDC-terra (DOI: 10.5880/enmap.2016.007) — not my own fieldwork; my contribution was the metadata handling, point–pixel matching, spectral preprocessing, modelling and validation.

**Before running:** the image data is not included in this repository. Download it from the DOI above, then replace the eight placeholder paths in `datasets` (`path/to/your/...`) with your own local paths.Python 3.10+, then pip install opencv-python scikit-image scikit-learn rasterio geopandas numpy pandas matplotlib scipy


## 1. RGB high-throughput leaf analysis → LAI & SPAD(0-high throughput leaf analysis random forest rgb.py)

**Task**: Estimate Leaf Area Index (LAI) and predict SPAD (chlorophyll proxy) using RGB imagery and a random forest model.
**Method**:
  Extracted vegetation indices (ExG, NGRDI, GLI) from soybean.jpg.
  Segmented the green canopy by thresholding min–max normalised ExG. The threshold was calibrated against a target green-coverage fraction (~80%), not fixed a     priori; on this image the selected threshold produced 88% canopy coverage. Morphological closing (5×5) and small-object removal (< 500 px) were applied for      display only — the reported coverage is computed on the raw thresholded mask.
  Converted coverage to LAI with a Beer–Lambert model, LAI = -ln(1 - f) / k, with an empirical extinction coefficient k = 0.5, giving LAI ≈ 4.23.
  Built a Random Forest regression model using 5 indices (ExG, NGRDI, GLI, VARI, TGI) to predict SPAD across 11 field treatments. Achieved an **R² of 0.455**.
  **Key insight**: Feature importance analysis identified VARI (Visible Atmospherically Resistant Index) as the most critical predictor.
  Caveat: N = 11 treatment-level means against 5 predictors. The R² is an exploratory, plot-level result and I do not treat its absolute value as a validated      accuracy; with 11 points the uncertainty is large. Its useful content is the ranking — which index carries the most chlorophyll signal — not the number itself.
  
## 2. Multi-date hyperspectral SPAD inversion(0-multiple time hyperspectral analysis new.py)

 **Task**: Predict SPAD from airborne hyperspectral imagery across multiple dates.
 **Method**:
   Extracted ground truth points from ESRI Shapefiles and matched them with hyperspectral pixels using GeoPandas and Rasterio.
   Used bands 36–90 of the 197 spectral bands (≈ 680–994 nm, red edge + NIR). Wavelengths follow the ENVI header's near-uniform ~6 nm sampling, anchored at band    36 = 680 nm; the header lists 477–1704 nm across 197 bands, with zenith/azimuth stored as bands 198–199.
   Applied Savitzky–Golay first-derivative preprocessing to enhance spectral features in the red-edge and NIR regions.
   Merged multi-date datasets and trained a Random Forest model via Leave-One-Out cross-validation. Achieved an **R² of 0.153** (N = 335).
 **Sensor note**: the imagery is **airborne AVIS-3 at 4 m ground sampling distance** (197 bands, 477–1704 nm), read from the ENVI header — not EnMAP satellite data, and not 30 m.
 **Key insight**: The modest accuracy is a problem of support rather than of regression. A 4 m pixel integrates about 16 m² of canopy, soil and shadow, whereas one SPAD reading covers roughly 6 mm² of a single leaf;  and the soil background was left unmasked. The band ranking reflects that entanglement: the red edge carries the chlorophyll signal, while the blue-green carries soil and residual atmosphere.

## Limitations / what I would do differently
Point–pixel matching is single-pixel and assumes the shapefile already shares the raster CRS (both are UTM 33N here). There is no to_crs guard and no 3×3 window, so a mis-projected shapefile would fail silently.
LOO CV does not group by date or plot, so samples from the same acquisition appear in both train and test. A GroupKFold by date would give a more honest — and lower — estimate.
No BRDF / illumination normalisation across the four dates, and soil background never masked from the spectra.
N = 11 in pipeline 1, so its R² is exploratory only.

## 🛠 Tech Stack

 **Languages**: Python
 **Libraries**: OpenCV, Scikit-learn, Rasterio, GeoPandas, NumPy, Pandas, Matplotlib, SciPy
