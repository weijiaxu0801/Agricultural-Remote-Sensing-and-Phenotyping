# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:25:17 2026

@author: asus
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

# Set font for plots (Optional, for English environment it's fine without)
plt.rcParams['axes.unicode_minus'] = False 

# ========== 1. Load Image ==========
img_path = 'soybean.jpg'  # Change to your image path
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ========== 2. Extract Channels ==========
R = img_rgb[:, :, 0].astype(float)
G = img_rgb[:, :, 1].astype(float)
B = img_rgb[:, :, 2].astype(float)

# ========== 3. Calculate Vegetation Indices ==========
eps = 1e-6
ExG = 2 * G - R - B
NGRDI = (G - R) / (G + R + eps)
GLI = (2 * G - R - B) / (2 * G + R + B + eps)

# ========== 4. Image Segmentation & LAI Estimation ==========
ExG_norm = (ExG - ExG.min()) / (ExG.max() - ExG.min() + eps)

# Find optimal threshold
best_threshold, best_ratio = None, None
target_ratio = 0.8
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    mask = ExG_norm > t
    ratio = np.sum(mask) / mask.size
    if best_ratio is None or abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
        best_ratio, best_threshold = ratio, t

binary_mask = ExG_norm > best_threshold
binary_mask = binary_mask & (G > R) & (G > B)
green_ratio = np.sum(binary_mask) / binary_mask.size

# Morphological operations for visualization
binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
kernel = np.ones((5, 5), np.uint8)
binary_mask_vis = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, kernel)
binary_mask_vis = binary_mask_vis > 0
binary_mask_vis = remove_small_objects(binary_mask_vis, min_size=500)

# Beer-Lambert Law for LAI
k = 0.5
LAI = -np.log(1 - green_ratio) / k if green_ratio < 1 else np.nan
chlorophyll_index = np.mean(NGRDI[binary_mask])
print(f"Green Coverage: {green_ratio:.3f}, Estimated LAI: {LAI:.3f}")

# ========== 5. Plot 6-Panel Analysis Figure ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img_rgb); axes[0, 0].set_title('Original RGB Image'); axes[0, 0].axis('off')
axes[0, 1].imshow(ExG_norm, cmap='RdYlGn'); axes[0, 1].set_title(f'ExG Index (Mean={np.mean(ExG):.3f})'); axes[0, 1].axis('off')
axes[0, 2].imshow(binary_mask_vis, cmap='gray'); axes[0, 2].set_title(f'Segmentation\nCoverage={green_ratio:.3f}, LAI={LAI:.2f}'); axes[0, 2].axis('off')
axes[1, 0].imshow(NGRDI, cmap='RdYlGn'); axes[1, 0].set_title(f'NGRDI Index (Mean={np.mean(NGRDI):.3f})'); axes[1, 0].axis('off')
axes[1, 1].imshow(GLI, cmap='RdYlGn'); axes[1, 1].set_title(f'GLI Index (Mean={np.mean(GLI):.3f})'); axes[1, 1].axis('off')

NGRDI_masked = np.where(binary_mask, NGRDI, np.nan)
im = axes[1, 2].imshow(NGRDI_masked, cmap='RdYlGn'); axes[1, 2].set_title(f'Canopy NGRDI (Mean={chlorophyll_index:.3f})'); axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

plt.tight_layout()
plt.savefig('rgb_analysis_result.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 6. Define 11 Treatment Regions ==========
regions = {
    1: (60, 30, 180, 160), 2: (50, 470, 220, 650), 3: (160, 350, 310, 480),
    4: (280, 180, 450, 350), 5: (420, 60, 590, 200), 6: (700, 60, 890, 230),
    7: (560, 350, 750, 500), 8: (520, 540, 680, 700), 9: (340, 600, 500, 740),
    10: (1060, 200, 1230, 380), 11: (900, 560, 1080, 700),
}
spad_values = {
    1: 23.9183, 2: 22.8157, 3: 25.2357, 4: 21.3028, 5: 26.9526, 6: 25.6397,
    7: 22.2432, 8: 24.3593, 9: 22.0909, 10: 25.0325, 11: 22.3167,
}

# ========== 7. Extract Indices for Each Region ==========
results = []
for num, (x1, y1, x2, y2) in regions.items():
    crop = img_rgb[y1:y2, x1:x2].astype(float)
    R, G, B = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]

    # Exclude orange marker pixels
    orange_mask = (R > 200) & (G < 150)
    valid_mask = ~orange_mask
    R_v, G_v, B_v = R[valid_mask], G[valid_mask], B[valid_mask]

    results.append({
        'Treatment': num,
        'ExG': np.mean(2 * G_v - R_v - B_v),
        'NGRDI': np.mean((G_v - R_v) / (G_v + R_v + eps)),
        'GLI': np.mean((2 * G_v - R_v - B_v) / (2 * G_v + R_v + B_v + eps)),
        'VARI': np.mean((G_v - R_v) / (G_v + R_v - B_v + eps)),
        'TGI': np.mean(G_v - 0.39 * R_v - 0.61 * B_v),
        'SPAD': spad_values[num]
    })

df = pd.DataFrame(results)

# ========== 8. Random Forest Modeling (LOO CV) ==========
X = df[['ExG', 'NGRDI', 'GLI', 'VARI', 'TGI']].values
y = df['SPAD'].values

rf = RandomForestRegressor(n_estimators=200, random_state=42)
y_pred = cross_val_predict(rf, X, y, cv=LeaveOneOut())

print(f"\nLOO CV R²: {r2_score(y, y_pred):.3f}")
print(f"LOO CV RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.3f}")

# ========== 9. Plot Random Forest Results ==========
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=['ExG', 'NGRDI', 'GLI', 'VARI', 'TGI'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y, y_pred, color='steelblue', s=80, edgecolors='k')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Measured SPAD')
axes[0].set_ylabel('Predicted SPAD')
axes[0].set_title(f'Random Forest (LOO CV)\nR² = {r2_score(y, y_pred):.3f}')

importances.sort_values().plot(kind='barh', ax=axes[1], color='darkgreen')
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance for SPAD Prediction')

plt.tight_layout()
plt.savefig('rf_spad_result.png', dpi=150, bbox_inches='tight')
plt.show()