# mdbrain & Adamchic et al. 2024 — Reference Notes

Notes from the paper "Artificial intelligence can help detecting incidental intracranial aneurysm on routine brain MRI using TOF MRA data sets" (Adamchic et al., Neuroradiology 2024) and research on mediaire's mdbrain software.

## The Paper (Adamchic et al. 2024)

### What they did
External validation of mdbrain (v4.7) on 500 patients. They did NOT train anything — they used mdbrain as an off-the-shelf commercial tool and compared its performance against a consultant neuroradiologist.

### Study cohort (500 patients)
- 500 consecutive patients from 3 academic hospitals in Berlin (after excluding 21 for poor quality/pathology)
- Mean age: 54 ± 14 years (range 19–95)
- Scanners: Siemens (Skyra 3T, Avanto/Aera 1.5T) and Philips (Achieva 1.5T)
  - 232 patients at 3T, 268 at 1.5T
- Slice thickness: 0.3×0.3×0.5 mm (3T) and 0.4×0.4×0.6 mm (1.5T)

### Aneurysm distribution in the study cohort
- 106 aneurysms in 85 patients (415 patients had no aneurysms)
- 104 saccular, 2 fusiform
- 68 patients with 1 aneurysm, 13 with 2, 4 with 3
- Mean size: 4.5 ± 5.3 mm (range 1–33 mm)
- Locations: MCA 22.6%, ICA 29.2%, AComA 20.8%, BA 13.2%, ACA 3.8%, PComA 3.8%, other 6.6%

### Results

| Reader | Aneurysms detected | Sensitivity |
|--------|-------------------|-------------|
| Neuroradiologist only | 98/106 | 92.5% |
| AI (mdbrain) only | 77/106 | 72.6% |
| Combined (consensus) | 106/106 | 100% (by definition — this was the reference standard) |

- AI reduced reading time by 19s (83.8s → 63.9s, 23% reduction)
- Neuroradiologist missed 8 aneurysms — significantly smaller (mean 2.7 ± 1.7 mm)
- AI missed 29 aneurysms — mean size 6.4 ± 9.3 mm, but 3 were giant thrombosed (30, 32, 33 mm); excluding those, mean missed = 3.5 ± 3.4 mm
- AI showed no size bias (unlike the neuroradiologist who missed small ones)
- Interobserver agreement (Cohen's Kappa): 0.94

---

## mdbrain (mediaire GmbH, Berlin)

### Product overview
- CE-marked Class IIb medical device (EU MDR)
- Available since January 2019, deployed in 11+ countries
- Runs on Linux/Ubuntu, supports cloud or local deployment
- Compatible with DICOM from Siemens, Philips, GE, Canon/Toshiba at 1.5T or 3T
- Processing time: 3–5 minutes per scan

### Modules
1. **Brain Volumetry** — dementia, MS, parkinsonian syndromes (input: 3D-T1-MPRAGE, 1.0×1.0 mm²)
2. **Lesion Characterization** — MS/dementia lesions (input: 3D-T1 + 3D-T2-FLAIR, 1.0×1.0 mm²)
3. **Aneurysm Detection** — introduced in v4.0 (May 2021) (input: TOF, 1.0×1.0 mm²)
4. **Tumor Differentiation** — glioma, metastasis, meningioma (input: multiple sequences)

### Aneurysm module — architecture and training

- **Task:** Segmentation (voxel-level prediction)
- **Architecture:** 3D U-NET CNN
- **Training data (v4.7):** 100 TOF-MRA subjects (mix of healthy and aneurysm), 93 saccular aneurysms
  - Scanners: Philips, at 1T, 1.5T, and 3T
  - No fusiform aneurysms; 4 (4.3%) partially thrombosed
  - Aneurysm locations: ICA C6 20%, ICA C7 22%, MCA M1/M2 20%, AComA 17%, ACA A2 10%, BA 9%
  - Exact split of healthy vs aneurysm patients not disclosed
  - Whether patients had multiple aneurysms not disclosed
- **Ground truth:** Binary masks of aneurysms, segmented by a radiologist
- **Preprocessing:** Resample to fixed spacing, normalize intensity to zero mean and unit variance
- **Training:** SGD, patch-based input with balanced sampling (some patches with aneurysm voxels, some without), on-the-fly augmentation
- A related Springer paper hints at 111 patients, sphere-based annotations, and 3D spline transform augmentation (possibly an earlier or different version of the training set)

### Performance across validation studies

| Study | Patients | Sensitivity | Specificity | PPV | NPV |
|-------|----------|-------------|-------------|-----|-----|
| Lehnen et al. 2022 (AJNR) | 191 | 72.6% | 87.2% | 67.9% | 88.5% |
| Adamchic et al. 2024 | 500 | 72.6% | — | — | — |

- Saccular aneurysms >5 mm (non-thrombosed): **100% sensitivity**
- Fusiform aneurysms: **33.3%** detection rate
- Thrombosed aneurysms: **16.7%** detection rate
- Both studies independently measured the same 72.6% overall sensitivity

### Known limitations
- Slice thickness must be below 1.5 mm
- No auto-recalibration on new data
- Poor on fusiform and thrombosed aneurysms (not in training data)
- Very small training set (100 subjects)

---

## Key Differences: mdbrain vs Our Project

| | mdbrain | Our project |
|---|---------|-------------|
| **Task** | Segmentation (voxel-level masks) | Classification (exam-level labels) |
| **Architecture** | 3D U-NET | Classification CNN (ResNet/DenseNet/custom) |
| **Ground truth** | Radiologist-segmented binary masks | Exam-level labels (aneurysm present/absent) |
| **Training data** | 100 subjects, Philips only | TBD |
| **Input** | TOF-MRA patches | TOF-MRA full volumes |
| **Output** | Segmentation mask (where is the aneurysm) | Binary prediction (is there an aneurysm) |

Segmentation requires expensive voxel-level annotations but provides spatial localization. Classification uses cheaper exam-level labels but gives no location info — less supervisory signal per sample, but much easier to scale the dataset.

---

## Sources
- [Adamchic et al. 2024 — Neuroradiology](https://link.springer.com/article/10.1007/s00234-024-03460-6)
- [Lehnen et al. 2022 — AJNR](https://www.ajnr.org/content/43/12/1700)
- [mediaire mdbrain product page](https://mediaire.ai/en/mdbrain/)
- [mediaire aneurysm announcement](https://mediaire.ai/en/mediaire-introduces-worlds-first-ki-tool-for-automatic-detection-of-aneurysms/)
- [Health AI Register — mdbrain](https://healthairegister.com/radiology/products/mediaire-mdbrain)
- [Springer — Efficient Data Strategy for Brain Aneurysm Detection](https://link.springer.com/chapter/10.1007/978-3-030-88210-5_22)
