# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:20:05 2026

@author: asus
"""

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


datasets = [ # change to your path
    (r'your path:\20120428_Neusling_AVIS',
     r'your path:\120428_AVIS_1.shp'),
    (r'your path:\20120525_Neusling_AVIS',
     r'your path:\120525_AVIS_2.shp'),
    (r'your path:\20120616_Neusling_AVIS',
     r'your path:\120616_AVIS_3.shp'),
    (r'your path:\20120908_Neusling_AVIS',
     r'your path:\120908_AVIS_4.shp'),
]


bands = list(range(36, 91))

all_spectra = []
all_spad = []

for img_path, shp_path in datasets:
    print(f"\nProcessing: {img_path.split(chr(92))[-1]}")

    with rasterio.open(img_path) as src:
        data = src.read(bands).astype(np.float32)
        transform = src.transform
        print(f"  Data shape: {data.shape}")


    data[data < 0] = np.nan
    data[data > 1.5] = np.nan
    data[~np.isfinite(data)] = np.nan


    for b in range(data.shape[0]):
        band = data[b]
        valid = band[np.isfinite(band)]
        if len(valid) > 0:
            band[np.isnan(band)] = np.nanmean(valid)
    data[~np.isfinite(data)] = 0

    print(f"  Cleaned range: {data.min():.3f} to {data.max():.3f}")


    gdf = gpd.read_file(shp_path)

    spad_col = None
    for col in gdf.columns:
        if 'spad' in col.lower() and 'std' not in col.lower() and 'sd' not in col.lower():
            spad_col = col
            break

    if spad_col is None:
        print(f"  Warning: SPAD column not found, skipping")
        continue

    count = 0
    for idx, row in gdf.iterrows():
        spad = row[spad_col]
        if pd.isna(spad) or spad <= 10:
            continue

        x, y = row.geometry.x, row.geometry.y
        row_pix, col_pix = rasterio.transform.rowcol(transform, x, y)

        if 0 <= row_pix < data.shape[1] and 0 <= col_pix < data.shape[2]:
            spectrum = data[:, row_pix, col_pix]
            if not np.all(np.isfinite(spectrum)):
                continue
            all_spectra.append(spectrum)
            all_spad.append(spad)
            count += 1

    print(f"  Valid points: {count}")

all_spectra = np.array(all_spectra)
all_spad = np.array(all_spad)

print(f"\nTotal merged samples: {len(all_spectra)}")
print(f"Spectra shape: {all_spectra.shape}")
print(f"SPAD range: {all_spad.min():.2f} to {all_spad.max():.2f}")


X_deriv = savgol_filter(all_spectra, window_length=11, polyorder=2, deriv=1, axis=1)
X_deriv[~np.isfinite(X_deriv)] = 0


rf = RandomForestRegressor(n_estimators=200, random_state=42)
y_pred = cross_val_predict(rf, X_deriv, all_spad, cv=LeaveOneOut())

print(f"\nLOO CV R²: {r2_score(all_spad, y_pred):.3f}")
print(f"LOO CV RMSE: {np.sqrt(mean_squared_error(all_spad, y_pred)):.3f}")


rf.fit(X_deriv, all_spad)
importances = rf.feature_importances_

top_bands = np.argsort(importances)[-10:][::-1]
print("\nTop 10 most important bands (Red Edge + NIR):")
for b in top_bands:
    actual_band = b + 36
    wavelength = 680 + (actual_band - 36) * 6
    print(f"  Band {actual_band} (~{wavelength}nm): Importance={importances[b]:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(all_spad, y_pred, color='steelblue', s=50, edgecolors='k', alpha=0.7)
axes[0].plot([all_spad.min(), all_spad.max()], [all_spad.min(), all_spad.max()], 'r--', lw=2)
axes[0].set_xlabel('Measured SPAD')
axes[0].set_ylabel('Predicted SPAD')
axes[0].set_title(f'Random Forest (LOO CV)\nR² = {r2_score(all_spad, y_pred):.3f}\nN = {len(all_spad)}')

axes[1].bar(range(len(importances)), importances, color='darkgreen')
axes[1].set_xlabel('Band Index (36-90, 680-994nm)')
axes[1].set_ylabel('Importance')
axes[1].set_title('Feature Importance in Red Edge + NIR')

plt.tight_layout()
plt.savefig('rf_multi_date_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nResult figure saved as rf_multi_date_result.png")
