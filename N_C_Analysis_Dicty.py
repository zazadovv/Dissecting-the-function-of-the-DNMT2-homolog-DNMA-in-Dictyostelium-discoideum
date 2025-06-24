import os
import numpy as np
import pandas as pd
from skimage import io, exposure, morphology, measure
from skimage.filters import gaussian, sobel
from skimage.restoration import denoise_bilateral
from skimage.morphology import disk, remove_small_objects, opening
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN
from tkinter import filedialog, Tk
import napari
from magicgui.widgets import FloatSlider, Slider, PushButton, Container
from tifffile import TiffFileError
from collections import defaultdict

# --- File selection ---
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select DAPI Image", filetypes=[("TIFF files", "*.tif")])
if not file_path:
    raise SystemExit("No file selected.")

# --- Load and normalize ---
try:
    img_rgb = io.imread(file_path)
    print(f"[✔] Loaded image: {file_path} | shape: {img_rgb.shape}")
except (TiffFileError, ValueError, OSError) as e:
    raise SystemExit(f"[✘] TIFF load failed: {e}")

if img_rgb.ndim == 2:
    img_dapi = img_rgb.astype(float)
else:
    img_dapi = img_rgb[..., 2].astype(float)

img_dapi -= img_dapi.min()
img_dapi /= (img_dapi.max() + 1e-8)
img_dapi = np.clip(img_dapi, 0, 1)
img_dapi = exposure.equalize_adapthist(img_dapi)
img_blur = gaussian(img_dapi, sigma=1)

# --- Napari viewer ---
viewer = napari.Viewer()
viewer.add_image(img_dapi, name="DAPI (Enhanced)")

# --- GUI sliders ---
nuc_thresh_slider = FloatSlider(label="Nucleus Threshold", min=0.1, max=1.0, step=0.01, value=0.5)
cyto_thresh_slider = FloatSlider(label="Cytoplasm Threshold", min=0.0, max=1.0, step=0.01, value=0.3)
sobel_weight_slider = FloatSlider(label="Edge Weight", min=0.0, max=1.0, step=0.05, value=0.5)
dbscan_eps_slider = Slider(label="DBSCAN ε (Nuclei Merge Distance)", min=5, max=100, step=1, value=30)
run_button = PushButton(label="Run Analysis")

def analyze():
    nuc_thresh = float(nuc_thresh_slider.value)
    cyto_thresh = float(cyto_thresh_slider.value)
    sobel_weight = float(sobel_weight_slider.value)
    dbscan_eps = int(dbscan_eps_slider.value)

    try:
        # --- Nucleus segmentation ---
        nuc_bin = img_blur > nuc_thresh
        nuc_bin = opening(nuc_bin, disk(2))
        nuc_bin = remove_small_objects(nuc_bin, min_size=50)
        nuc_bin = ndi.binary_fill_holes(nuc_bin)

        distance = ndi.distance_transform_edt(nuc_bin)
        local_maxi = peak_local_max(distance, labels=nuc_bin, footprint=np.ones((15, 15)), exclude_border=False)
        if local_maxi.size == 0:
            viewer.text_overlay.text = "No nuclei detected."
            return

        clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(local_maxi)
        group_labels = clustering.labels_
        grouped_markers = np.zeros_like(nuc_bin, dtype=np.int32)
        for group_id in np.unique(group_labels):
            coords = local_maxi[group_labels == group_id]
            for y, x in coords:
                grouped_markers[y, x] = group_id + 1

        nucleus_label = watershed(-distance, grouped_markers, mask=nuc_bin)

        # --- Enhanced cytoplasm segmentation ---
        dapi_denoised = denoise_bilateral(img_dapi, sigma_color=0.05, sigma_spatial=5, channel_axis=None)
        edge_mag = sobel(dapi_denoised)

        cyto_candidate = dapi_denoised > cyto_thresh
        cyto_candidate = ndi.binary_fill_holes(cyto_candidate)
        cyto_candidate = morphology.remove_small_objects(cyto_candidate, min_size=100)
        cyto_candidate = morphology.binary_opening(cyto_candidate, morphology.disk(2))

        fusion = (1.0 - dapi_denoised) * (1 - sobel_weight) + edge_mag * sobel_weight
        fusion[~cyto_candidate] = 1.0

        seed_markers = morphology.dilation(nucleus_label, morphology.disk(1))
        fusion[img_dapi > 0.98] = 1.0
        expanded_label = watershed(fusion, seed_markers, mask=cyto_candidate)
        cytoplasm_label = expanded_label.copy()
        cytoplasm_label[nucleus_label > 0] = 0

        # --- Merge cytoplasm regions that share nuclei ---
        nucleus_label_dilated = morphology.dilation(nucleus_label, morphology.disk(1))
        region_to_nuclei = {}
        for cyto in measure.regionprops(cytoplasm_label):
            cyto_id = cyto.label
            cyto_mask = cytoplasm_label == cyto_id
            overlapping_nuclei = np.unique(nucleus_label_dilated[cyto_mask])
            overlapping_nuclei = tuple(sorted(overlapping_nuclei[overlapping_nuclei > 0]))
            if overlapping_nuclei:
                region_to_nuclei[cyto_id] = overlapping_nuclei

        merge_map = defaultdict(list)
        for rid, nuclei in region_to_nuclei.items():
            merge_map[nuclei].append(rid)

        merged_cytoplasm = np.zeros_like(cytoplasm_label)
        for new_id, (nuc_group, region_ids) in enumerate(merge_map.items(), start=1):
            merged_mask = np.isin(cytoplasm_label, region_ids)
            merged_cytoplasm[merged_mask] = new_id

        cytoplasm_label = merged_cytoplasm

        # --- Measure properties ---
        results = []
        for cyto in measure.regionprops(cytoplasm_label):
            cyto_id = cyto.label
            cyto_mask = cytoplasm_label == cyto_id
            cyto_area = np.sum(cyto_mask)
            if cyto_area < 10:
                continue

            overlapping_nuclei = np.unique(nucleus_label_dilated[cyto_mask])
            overlapping_nuclei = overlapping_nuclei[overlapping_nuclei > 0]
            if len(overlapping_nuclei) == 0:
                print(f"[!] Skipping Cell_ID {cyto_id}: no nucleus overlap")
                continue

            nuc_area = sum(np.sum(nucleus_label == nid) for nid in overlapping_nuclei)
            ratio = nuc_area / cyto_area if cyto_area > 0 else np.nan
            if np.isnan(ratio):
                print(f"[!] Invalid area ratio for Cell_ID {cyto_id}: nuc_area={nuc_area}, cyto_area={cyto_area}")

            results.append({
                "Cell_ID": cyto_id,
                "Num_Nuclei": len(overlapping_nuclei),
                "Nucleus_Area_Total": nuc_area,
                "Cytoplasm_Area": cyto_area,
                "Area_Ratio": ratio
            })

        df = pd.DataFrame(results)
        out_csv = os.path.splitext(file_path)[0] + "_nuc_cyto_area_ratio_FINAL.csv"
        df.to_csv(out_csv, index=False)
        print(f"[✔] Saved: {out_csv}")

        for lname in ["Nuclei", "Cytoplasm", "Nucleus Outline"]:
            if lname in viewer.layers:
                viewer.layers.remove(lname)

        viewer.add_labels(nucleus_label, name="Nuclei")
        viewer.add_labels(cytoplasm_label, name="Cytoplasm")
        outline = find_boundaries(nucleus_label, mode="outer").astype(np.uint8)
        viewer.add_image(outline, name="Nucleus Outline", colormap="red", blending="additive", contrast_limits=(0, 1))

        if not df.empty:
            viewer.text_overlay.text = f"{len(df)} cells | Mean Area Ratio: {df['Area_Ratio'].mean():.2f}"
        else:
            viewer.text_overlay.text = "Detected cells, but all area ratios were invalid."

    except Exception as e:
        print(f"[✘] Error during analysis: {e}")
        viewer.text_overlay.text = f"Error: {e}"

run_button.changed.connect(analyze)
gui_panel = Container(widgets=[nuc_thresh_slider, cyto_thresh_slider, sobel_weight_slider, dbscan_eps_slider, run_button])
viewer.window.add_dock_widget(gui_panel, area="right")
viewer.text_overlay.visible = True
napari.run()