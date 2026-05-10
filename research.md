# Research Keywords & Literature Search Strategy

When searching databases like Scopus for state-of-the-art methods in aneurysm classification using machine learning or deep learning, it helps to combine keywords from different categories. Since this project is named `aneurysm_cnn`, I have tailored these to include deep learning and Convolutional Neural Networks (CNNs).

## Keyword Categories

### 1. Medical / Domain Terms
*   "aneurysm" (Note: This will catch *all* types of aneurysms, which is great for a broad search, but it will also include abdominal/aortic aneurysms. If you get too many irrelevant results, use the terms below to narrow it down).
*   "intracranial aneurysm" OR "cerebral aneurysm" OR "brain aneurysm" (Use these if you want to filter out non-brain aneurysms).
*   "aortic aneurysm" (If you are looking for non-brain aneurysms).

### 2. Task / Goal Terms
*   "classification"
*   "detection"
*   "segmentation" (often a prerequisite or concurrent task with classification)
*   "rupture risk" OR "rupture prediction" (common classification targets)
*   "diagnosis"

### 3. Technology / Method Terms
*   "machine learning" OR "ML"
*   "deep learning" OR "DL"
*   "convolutional neural network" OR "CNN" OR "3D CNN"
*   "artificial intelligence" OR "AI"
*   "computer vision"

### 4. Imaging Modalities (Highly relevant for CNNs)
*   "CTA" OR "computed tomography angiography"
*   "MRA" OR "magnetic resonance angiography"
*   "DSA" OR "digital subtraction angiography"
*   "3D imaging"

---

## Recommended Scopus Search Strings

You can copy and paste these query strings directly into Scopus's Advanced Search or combine them in the basic search fields.

**1. Broad Search (Good starting point):**
```
TITLE-ABS-KEY ( ("aneurysm") AND ("classification" OR "detection") AND ("machine learning" OR "deep learning" OR "convolutional neural network" OR "CNN") )
```

**2. Focused strictly on Brain Aneurysms & Deep Learning:**
```
TITLE-ABS-KEY ( ("cerebral aneurysm" OR "intracranial aneurysm") AND ("deep learning" OR "CNN" OR "convolutional neural network") AND ("classification" OR "detection" OR "segmentation") )
```

**3. Focused on specific Imaging Modalities and CNNs:**
```
TITLE-ABS-KEY ( ("aneurysm") AND ("convolutional neural network" OR "CNN" OR "3D CNN") AND ("MRA" OR "CTA" OR "angiography") )
```

**4. Focused on Predicting Rupture (A very common classification task):**
```
TITLE-ABS-KEY ( ("aneurysm") AND ("rupture" AND "prediction") AND ("machine learning" OR "deep learning") )
```

## Tips for your Literature Review
*   **Filter by Year:** Limit your search to the last 3-5 years (e.g., 2020-2026) to see the true "state-of-the-art" as deep learning moves very fast.
*   **Look for Reviews:** Add `AND ( LIMIT-TO ( DOCTYPE , "re" ) )` to find Review papers. Review papers are fantastic for getting a summary of all recent methods in one place.
*   **Check the Metrics:** Look at the performance metrics they use (e.g., AUC, Sensitivity, Specificity, F1-score) to know what target numbers you should aim for in your own `training_engine`.
