# Justification & Diagnostic Scripts

Research scripts that document **why** specific design decisions were made.
These are not part of the main pipeline — they are standalone visualisation
tools run manually to produce figures for the thesis.

---

## Data Engine — `data_engine/scripts/justification/`

### justify_centroid_crop.py

**Decision:** Crop the XY field of view centered on the tissue centroid, not
the geometric center of the volume.

**What it shows:** Side-by-side overlay of centroid vs geometric-center crop
boxes on mid-axial slices. The centroid crop consistently keeps the Circle
of Willis (where aneurysms occur) centered, while geometric-center crop
clips it in volumes with off-center anatomy.

### justify_cube_shape.py

**Decision:** Use 128x128x128 cubes, not 128x128x96 (shorter Z).

**What it shows:** Renders the top-N and bottom-N axial slices of converted
volumes. Anatomy (vessels, brain tissue) is present at the Z extremes —
cropping to 96 slices would discard diagnostically relevant regions.

### justify_isotropic_choice.py

**Decision:** Resample to isotropic spacing rather than preserving native
anisotropic spacing.

**What it shows:** 6-panel comparison of three Z-axis strategies (isotropic,
shrink-to-target, native 1:1) with histograms of Z spacing distributions.
Isotropic resampling produces uniform voxel geometry with acceptable
interpolation cost.

---

## Training Engine — `training_engine/scripts/diagnostics/`

### view_augmentation.py

**Purpose:** Visual validation that the augmentation pipeline produces
plausible training variants.

**What it shows:** For each sample: original + 3 augmented versions, rendered
as 3 orthogonal mid-slices (axial, coronal, sagittal). Confirms flips,
rotations, and noise produce realistic-looking volumes.

### view_normalization.py

**Purpose:** Compare intensity normalisation strategies across different
scanners and field strengths.

**What it shows:** Overlaid histograms (two scanners per pair) and mid-slice
images, before and after each normalisation method (original, percentile,
z-score, histogram matching). Used to select the normalisation strategy
that best reduces inter-scanner intensity variation.

---

## Data Engine — `data_engine/scripts/`

### build_histogram_template.py

**Purpose:** Build a reference histogram template for histogram matching
normalisation.

**What it does:** Loads NIfTI volumes from a reference scanner (Skyra 3T),
computes the average quantile distribution across all reference exams, and
saves it as a `.npy` template file used by the training engine's
`HistogramMatched` transform.
