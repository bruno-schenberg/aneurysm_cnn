"""Shared 3D Frangi vesselness computation.

Used by generate_frangi_mips.py (--mode 3d) and generate_vessel_crop.py —
both need the same GPU-first, sigmas=range(1,5) vesselness volume.
"""
import numpy as np


def compute_vesselness_3d(
    data_norm: np.ndarray, sigmas=range(1, 5), *, cpu_fallback: bool = True,
) -> np.ndarray:
    """3D Frangi vesselness on a normalized (~[0,1]) volume.

    cuCIM (GPU) by default. If cpu_fallback is True (default), falls back
    to skimage (CPU) when cupy/cucim aren't importable — full-volume 3D
    Frangi on CPU is impractically slow at production scale (minutes per
    volume x hundreds of volumes), so this is a "still runs, just slow"
    escape hatch, not meant for real runs. If cpu_fallback is False, a
    missing cupy/cucim raises ImportError instead of silently falling back
    to that much slower path.
    """
    try:
        import cupy as cp
        from cucim.skimage.filters import frangi as cucim_frangi
    except ImportError:
        if not cpu_fallback:
            raise
        from skimage.filters import frangi as skimage_frangi

        return skimage_frangi(data_norm, sigmas=sigmas, black_ridges=False).astype(np.float32)

    data_gpu = cp.asarray(data_norm)
    vesselness_gpu = cucim_frangi(data_gpu, sigmas=sigmas, black_ridges=False)
    vesselness = cp.asnumpy(vesselness_gpu).astype(np.float32)
    del data_gpu, vesselness_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return vesselness
