# Aneurysm Size Reference

Original resolution: 768×768 pixels over 180mm FOV → **0.234 mm/pixel**

Resampled variants:
- `iso_128`: 128³ voxels at 1.40625 mm/vox
- `iso_256`: 256³ voxels at 0.703125 mm/vox

| Aneurysm size | Category | px @ 768×768 / 180mm | vox @ iso_128 (1.406 mm) | vox @ iso_256 (0.703 mm) |
|---|---|---|---|---|
| 2mm | tiny | 8.5 | 1.4 | 2.8 |
| 3mm | tiny | 12.8 | 2.1 | 4.3 |
| 4mm | small | 17.1 | 2.8 | 5.7 |
| 5mm | small | 21.3 | 3.6 | 7.1 |
| 7mm | small | 29.9 | 5.0 | 10.0 |
| 10mm | medium | 42.7 | 7.1 | 14.2 |
| 15mm | medium | 64.0 | 10.7 | 21.3 |
| 25mm | large | 106.7 | 17.8 | 35.6 |

At `iso_128`, aneurysms under 5mm are 1–3 voxels wide — effectively undetectable.
At `iso_256`, a 3mm aneurysm is ~4 voxels, which is marginal but workable.
